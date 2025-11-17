python -m src.create_speaker \
    --model-name-or-path /home/anhnct/project/sparktts_vllm_server/resources/extend_vocab_pretrained \
    --speaker-names female_happy_english \
    --prompt-speech-paths /home/anhnct/project/Spark-TTS-finetune/DATA/elevenlab_dataset/wavs/32499.wav \
    --prompt-texts "i feel so happy <chuckle> when i hear my favorite song, it makes me want to dance <laugh>" \
    --output-path /home/anhnct/project/sparktts_vllm_server/resources/speakers
