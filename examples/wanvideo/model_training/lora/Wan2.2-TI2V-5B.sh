nohup accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/track \
  --dataset_metadata_path data/track/config/metadata.csv \
  --data_file_keys "video,flow_line" \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 25 \
  --remove_prefix_in_ckpt "pipe.dit.,pipe.flow_line_adapter." \
  --trainable_models "flow_line_adapter" \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 256 \
  --lora_checkpoint "models/lora/lora_5B.safetensors" \
  --extra_inputs "input_image,flow_line" \
> train.log 2>&1 &
