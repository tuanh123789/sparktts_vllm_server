sudo docker run --gpus all \
    -v /home/anhnct/project/sparktts_vllm_server/resources/extend_vocab_pretrained_bilingual/LLM:/sparktts \
    -p 9020:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /sparktts \
    --gpu-memory-utilization 0.6