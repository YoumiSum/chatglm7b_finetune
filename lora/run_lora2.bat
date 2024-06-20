python finetune_lora.py ^
    --dataset_path data/weather_token ^
    --lora_rank 4 ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 1 ^
    --num_train_epochs 50 ^
    --save_steps 1000 ^
    --save_total_limit 2 ^
    --learning_rate 1e-4 ^
    --remove_unused_columns false ^
    --logging_steps 1 ^
    --output_dir weather-lora ^
    --logging_dir ./tf_logs
    