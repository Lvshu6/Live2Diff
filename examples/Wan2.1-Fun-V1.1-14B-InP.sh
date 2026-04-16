accelerate launch \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/track2np \
  --dataset_metadata_path data/track2np/metadata.csv \
  --data_file_keys "video" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-InP:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-InP:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 25 \
  --remove_prefix_in_ckpt "pipe.flow_line_adapter." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-InP_full_split_cache" \
  --extra_inputs "input_image,track" \
  --task "sft:data_process" \
  --offload_models "PAI/Wan2.1-Fun-V1.1-14B-InP:diffusion_pytorch_model*.safetensors" 

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml
accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path ./models/train/Wan2.1-Fun-V1.1-14B-InP_full_split_cache \
  --data_file_keys "video,flow_line" \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-InP:diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.flow_line_adapter." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-InP_full_split" \
  --trainable_models "flow_line_adapter" \
  --extra_inputs "input_image,flow_line" \
  --task "sft:train" \
  --initialize_model_on_cpu 