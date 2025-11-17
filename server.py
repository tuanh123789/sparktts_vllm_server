import io
import os
import re
import pickle
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import soundfile as sf
from typing import Optional
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Import tokenizer from sparktts
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import TASK_TOKEN_MAP

# Adjust these paths as needed (loaded from .env)
SPEAKER_DIR = os.getenv("SPEAKER_DIR")
AUDIO_TOKENIZER_PATH = os.getenv("AUDIO_TOKENIZER_PATH")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load resources once at startup ---
print("Loading audio tokenizer...")
audio_tokenizer = BiCodecTokenizer(AUDIO_TOKENIZER_PATH, device=device)

print("Loading speakers...")
speakers = {}
if not os.path.isdir(SPEAKER_DIR):
    raise RuntimeError(f"Speaker dir not found: {SPEAKER_DIR}")
for speaker in os.listdir(SPEAKER_DIR):
    if not speaker.endswith('.pkl'):
        continue
    with open(os.path.join(SPEAKER_DIR, speaker), 'rb') as f:
        speakers[speaker.replace('.pkl', '')] = pickle.load(f)

# --- FastAPI app ---
app = FastAPI(title="SparkTTS FastAPI Server")

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = "elevenlab"

def synthesize_wav(text: str, speaker_name: str,
                   max_tokens: int = 3000,
                   temperature: float = 0.8,
                   top_p: float = 0.95,
                   top_k: int = 50):
    """
    Run the same inference pipeline as the original script and return path to a wav file.
    This function is synchronous and will block the worker while calling the vLLM HTTP endpoint.
    """
    if speaker_name not in speakers:
        raise ValueError(f"Unknown speaker: {speaker_name}")

    speaker_info = speakers[speaker_name]
    prompt_text = speaker_info["prompt_text"]
    global_tokens = speaker_info["global_tokens"]
    semantic_tokens = speaker_info["semantic_tokens"]
    global_token_ids = speaker_info.get("global_token_ids")

    # Prepend comma as in original script
    text = ", " + text

    inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
        prompt_text,
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
    ]
    prompt = "".join(inputs)

    payload = {
        "model": "/sparktts",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Call vLLM server
    resp = requests.post(VLLM_ENDPOINT, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"vLLM server returned {resp.status_code}: {resp.text}")

    output_tokens = resp.json()["choices"][0]["text"]

    ids = re.findall(r"bicodec_semantic_(\d+)", output_tokens)
    if len(ids) == 0:
        raise RuntimeError("No semantic tokens found in vLLM output")

    pred_semantic_ids = (
        torch.tensor([int(token) for token in ids]).long().unsqueeze(0)
    )

    # Detokenize to waveform
    wav = audio_tokenizer.detokenize(
        torch.tensor(global_token_ids).to(device).squeeze(0),
        pred_semantic_ids.to(device),
    )

    return wav

def preprocess_text(text):
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("?", ",")
    text = text.replace("!", ",")
    text = text.lower()

    return text


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    """Synthesize text to a WAV file and return it as a file response."""
    try:
        norm_text = preprocess_text(req.text)
        wav = synthesize_wav(norm_text, req.speaker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format='WAV')
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")

@app.post("/long_synthesize")
def long_synthesize(req: TTSRequest):
    """Synthesize text to a WAV file and return it as a file response."""
    try:
        # --- 1. Split text into sentences ---
        sentences = req.text.split(". ")
        if not sentences:
            raise ValueError("Input text contains no valid sentences.")

        wav_chunks = []

        # --- 2. Process each sentence ---
        for sent in sentences:
            norm_text = preprocess_text(sent)
            wav = synthesize_wav(norm_text, req.speaker)
            wav_chunks.append(wav)

        # --- 3. Concatenate all audio ---
        final_wav = np.concatenate(wav_chunks)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # --- 4. Export to BytesIO ---
    buf = io.BytesIO()
    sf.write(buf, final_wav, 16000, format='WAV')
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
