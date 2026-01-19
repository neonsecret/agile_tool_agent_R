accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 train.py
wandb login
huggingface-cli login