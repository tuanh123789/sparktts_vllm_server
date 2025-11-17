import os
import pickle
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


class Speaker:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_name_or_path}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(model_name_or_path)

    def create_speaker_info(self, prompt_speech_path, prompt_text):
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )

        speaker_info = {
            "prompt_text": prompt_text,
            "global_tokens": global_tokens,
            "semantic_tokens": semantic_tokens,
            "global_token_ids": global_token_ids
        }

        return speaker_info

    def create_speaker(self, speaker_names, prompt_speech_paths, prompt_texts, output_path):
        for speaker_name, prompt_speech_path, prompt_text in tqdm(
            zip(speaker_names, prompt_speech_paths, prompt_texts)
        ):
            speaker_info = self.create_speaker_info(
                prompt_speech_path=prompt_speech_path,
                prompt_text=prompt_text
            )

            with open(os.path.join(output_path, f"{speaker_name}.pkl"), "wb") as f:
                pickle.dump(speaker_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create speaker pickle files.")

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Path to the model folder."
    )

    parser.add_argument(
        "--speaker-names",
        type=str,
        nargs="+",
        required=True,
        help="List of speaker names."
    )

    parser.add_argument(
        "--prompt-speech-paths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to the prompt speech audio files."
    )

    parser.add_argument(
        "--prompt-texts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompt texts corresponding to each speaker."
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Folder to save generated speaker .pkl files."
    )

    args = parser.parse_args()

    sp = Speaker(args.model_name_or_path)

    sp.create_speaker(
        speaker_names=args.speaker_names,
        prompt_speech_paths=args.prompt_speech_paths,
        prompt_texts=args.prompt_texts,
        output_path=args.output_path
    )
