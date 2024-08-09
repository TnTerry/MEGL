# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train_cnn_megl_trainer.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --train_type use_lora \
    --data_path /root/MEGL/Datasets/Object_Classification \
    --transformation_type HAICS \
    --precision_type bf16 \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    --output_dir output_model \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 5e-5 \
    --logging_steps 10 \
