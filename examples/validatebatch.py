import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from pathlib import Path
import os

# ================== 配置 ==================
DEVISE = "cuda:0"
input_folder = Path("data/longtest")          # 原始视频所在目录
flow_base = input_folder             # flow 的根目录（与 input_folder 同级）
flow_subdir = Path("flow_line")            # flow 子目录名
output_folder = Path("valid/longtest")              # 输出目录

# 确保输出目录存在
output_folder.mkdir(exist_ok=True,parents=True)

video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

# ================== 加载模型 ==================
print("Loading model...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVISE,
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    vram_limit=16.0
)
pipe.vram_management_enabled = True

# 加载 flow_line_adapter 微调权重
#"WanFlow/Wan2.1-Fun-V1.1-14B-InP/full/epoch-5.safetensors"
flow_state_dict = load_state_dict("WanFlow/Wan2.1-Fun-V1.1-14B-InP/full/0309/epoch-3.safetensors")
pipe.flow_line_adapter.load_state_dict(flow_state_dict)
pipe.flow_line_adapter.to(DEVISE)  # 确保 adapter 在正确设备上

print("Model loaded successfully.")

# ================== 批量处理 ==================
for video_path in input_folder.iterdir():
    if video_path.suffix.lower() not in video_extensions:
        continue

    print(f"\nProcessing: {video_path.name}")

    # 构造对应的 flow 路径：base / flow / video_name
    flow_path = flow_base / flow_subdir / video_path.name
    if not flow_path.exists():
        print(f"⚠️ Flow file not found: {flow_path}, skipping.")
        continue

    try:
        # 加载视频和光流数据
        video_data = VideoData(video_path)
        flow_data = VideoData(flow_path)

        if len(video_data) == 0 or len(flow_data) == 0:
            print(f"⚠️ Empty video or flow, skipping: {video_path.name}")
            continue

        img_width, img_height = video_data[0].size
        num_frames = len(video_data)

        # 推理生成视频
        generated_video = pipe(
            prompt="move",
            negative_prompt="",
            input_image=video_data[0],
            # end_image=video_data[-1],
            flow_line=flow_data,
            seed=0,
            tiled=True,
            height=img_height,
            width=img_width,
            num_inference_steps=20,
            num_frames=num_frames
        )

        # 保存结果
        output_path = output_folder / video_path.name
        save_video(generated_video, output_path, fps=15, quality=5)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {e}")
        continue

print("\n🎉 Batch processing completed.")