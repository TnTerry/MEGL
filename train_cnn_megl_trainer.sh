# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train_cnn_megl_trainer.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --train_type use_lora \
    --data_path MEGL/Datasets/Object_Classification \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    --output_dir output_model \
    --num_train_epochs 5 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 5e-5 \
    --logging_steps 10
