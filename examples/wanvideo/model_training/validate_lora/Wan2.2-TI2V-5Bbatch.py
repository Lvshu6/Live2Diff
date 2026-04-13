import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from pathlib import Path
from diffsynth.utils.yxd.remove import remove_prefix_from_ckpt
    
# ================== 配置路径 ==================
input_folder = Path("data/test")          # 原始视频所在目录
flow_base = input_folder           # flow 的根目录（与你原代码一致）
flow_subdir = Path("flow_line")                 # flow 子目录名
output_folder = Path("valid/5B/0121-2219")                   # 输出目录
lora_path = "models/train/Wan2.2-TI2V-5B_lora/epoch-0.safetensors"
remove_prefix_from_ckpt(lora_path, lora_path, ["pipe.flow_line_adapter.", "pipe.dit."])
# 确保输出目录存在
output_folder.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
# ================== 加载模型 ==================
print("Loading model...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
    vram_limit=30.0
)
pipe.vram_management_enabled = True

# 加载 LoRA 和 flow adapter
state_dict = load_state_dict(lora_path)
lora_state_dict = {}
flow_adapter_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("flow_line_patch_embedding.") or key.startswith("flow_line_blocks."):
        flow_adapter_state_dict[key] = value
    else:
        lora_state_dict[key] = value

pipe.flow_line_adapter.load_state_dict(flow_adapter_state_dict)
pipe.flow_line_adapter.to(DEVICE)
pipe.load_lora(pipe.dit, alpha=1, state_dict=lora_state_dict)
print("Model loaded.")

# ================== 批量处理 ==================
video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # 支持的视频格式

for video_path in input_folder.iterdir():
    if video_path.suffix.lower() not in video_extensions:
        continue

    print(f"\nProcessing: {video_path.name}")

    # 构造 flow 路径：base / flow / video_name
    flow_path = flow_base / flow_subdir / video_path.name
    if not flow_path.exists():
        print(f"⚠️ Flow file not found: {flow_path}, skipping.")
        continue

    try:
        # 加载视频和 flow
        video_data = VideoData(video_path)
        flow_data = VideoData(flow_path)

        if len(video_data) == 0 or len(flow_data) == 0:
            print(f"⚠️ Empty video or flow, skipping: {video_path.name}")
            continue

        img_width, img_height = video_data[0].size
        num_frames = len(video_data)

        # 推理
        generated_video = pipe(
            prompt="move",
            negative_prompt="",
            input_image=video_data[0],
            flow_line=flow_data,
            seed=0,
            tiled=True,
            height=img_height,
            width=img_width,
            num_inference_steps=20,
            num_frames=num_frames
        )

        # 保存
        output_path = output_folder / video_path.name
        save_video(generated_video, output_path, fps=15, quality=5)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {e}")
        continue

print("\n🎉 Batch processing completed.")


