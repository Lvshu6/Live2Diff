import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import math
import torch
from cotracker.utils.PureVisualizer import PureVisualizer
import torch.nn.functional as F
from diffsynth.utils.data import save_video, VideoData
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from pathlib import Path
import os

# ==================== 全局变量与配置 ====================
Model = None
WanPipe = None
DEFAULT_DEVICE = "cuda"
SELECT_RADIUS = 7
BASE_DIR = "gradio_outputs"
# ==================== 通用工具函数 ====================
def get_unique_path(input_path):
    base_path = Path(input_path).absolute()
    parent_dir = base_path.parent
    file_name = base_path.name
    name, ext = os.path.splitext(file_name)
    num = 1
    while True:
        new_file_name = f"{name}_{num:05d}{ext}"
        new_path = parent_dir / new_file_name
        if not new_path.exists():
            return new_path
        num += 1

def load_wan_pipe():
    path = "WanFlow/Wan2.1-Fun-V1.1-14B-InP/full/0315/epoch-6.safetensors"
    print("正在加载 WanFlow 模型...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=DEFAULT_DEVICE,
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        vram_limit=40.0
    )
    pipe.vram_management_enabled = True
    flow_state_dict = load_state_dict(path)
    pipe.flow_line_adapter.load_state_dict(flow_state_dict)
    pipe.flow_line_adapter.to(DEFAULT_DEVICE)
    print("WanFlow 模型加载完成。\n")
    return pipe

def visualize(pred_tracks, canvas_size, filename="video", temp=True):
    save_dir = Path(BASE_DIR)
    track_dir= save_dir / "track"
    flow_dir= save_dir / "flow_line"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    filename = Path(filename +".mp4")
    path = flow_dir / filename
    if not temp:
        path = get_unique_path(path)
        path=Path(path)
    track_path=track_dir / path.with_suffix(".npy").name
    filename = path.stem
    visualizer = PureVisualizer(save_dir=Path(path).parent, linewidth=3, fps=10, history=3, bg_color=(255, 255, 255))
    visualizer.visualize(tracks=pred_tracks, filename=filename, save_video=True, canvas_size=canvas_size)
    
    tensor_clone = pred_tracks.clone().cpu().detach()
    np_data = tensor_clone.numpy()
    np.save(track_path, np_data)
    return path

def wan_flow(img, flow_path):
    global WanPipe
    if WanPipe is None:
        WanPipe = load_wan_pipe()
    flow = VideoData(flow_path)
    img_width, img_height = img.size
    video = WanPipe(
        prompt="move", negative_prompt="", input_image=img, flow_line=flow,
        seed=0, tiled=True, height=img_height, width=img_width, 
        num_inference_steps=20, num_frames=len(flow)
    )
    path = Path(BASE_DIR) /"output"/ "video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path = get_unique_path(path.as_posix())
    save_video(video, path, fps=15, quality=5)
    return path.as_posix()

# ==================== 轨迹绘制与管理逻辑 ====================
def draw_trajectories_on_image(original_image_pil, trajectories, current_traj_idx, temp_point=None):
    if original_image_pil is None:
        return None
        
    image = original_image_pil.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    for traj_idx, trajectory in enumerate(trajectories):
        if len(trajectory) >= 2:
            for i in range(len(trajectory) - 1):
                start_point = tuple(map(int, trajectory[i]))
                end_point = tuple(map(int, trajectory[i+1]))
                line_color = (0, 255, 0, 200) if traj_idx == current_traj_idx else (0, 200, 0, 100)
                draw.line([start_point, end_point], fill=line_color, width=3)
                
        for point_idx, point in enumerate(trajectory):
            x, y = int(point[0]), int(point[1])
            if traj_idx == current_traj_idx:
                if point_idx == 0:
                    color = (255, 255, 0, 255)
                elif point_idx == len(trajectory) - 1:
                    color = (0, 0, 255, 255)
                else:
                    color = (255, 0, 0, 255)
            else:
                color = (128, 128, 128, 180)
                
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(0, 0, 0, 255), width=1)
    
    if temp_point:
        x, y = int(temp_point[0]), int(temp_point[1])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 0, 150), outline=(0, 0, 0, 255), width=1)
    
    combined = Image.alpha_composite(image, overlay).convert("RGB")
    return combined

def convert_trajectories_to_tensor(trajectories, num_frames):
    trajectories = [traj for traj in trajectories if len(traj) > 0]
    if not trajectories:
        return torch.zeros(1, num_frames, 0, 2)
    max_trajs = len(trajectories)
    tensor = torch.zeros(1, num_frames, max_trajs, 2)
    
    for traj_idx, trajectory in enumerate(trajectories):
        point_count = len(trajectory)
        if point_count == 0:
            continue
        shake = 0.5
        if point_count == 1:
            x0, y0 = trajectory[0]
            tensor[0, 0, traj_idx, 0] = float(x0)
            tensor[0, 0, traj_idx, 1] = float(y0)
            for t in range(1, num_frames):
                rand_x = x0 + np.random.uniform(-shake, shake)
                rand_y = y0 + np.random.uniform(-shake, shake)
                tensor[0, t, traj_idx, 0] = float(rand_x)
                tensor[0, t, traj_idx, 1] = float(rand_y)
        else:
            traj_np = np.array(trajectory, dtype=np.float32)
            N = len(traj_np)
            original_times = np.linspace(0, N-1, N)
            target_times = np.linspace(0, N-1, num_frames)
            x_interp = np.interp(target_times, original_times, traj_np[:, 0])
            y_interp = np.interp(target_times, original_times, traj_np[:, 1])
            for t in range(num_frames):
                tensor[0, t, traj_idx, 0] = x_interp[t]
                tensor[0, t, traj_idx, 1] = y_interp[t]
    return tensor

def handle_click_new(original_image, trajectories, current_traj_idx, temp_point, evt: gr.SelectData):
    if original_image is None:
        return None, trajectories, current_traj_idx, temp_point
    
    W, H = original_image.size
    x_click, y_click = evt.index
    x, y = int(x_click), int(y_click)
    
    if not (0 <= x < W and 0 <= y < H):
        updated_img = draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point)
        return updated_img, trajectories, current_traj_idx, temp_point
    
    closest_traj_idx = -1
    min_dist = float('inf')
    
    for traj_idx, trajectory in enumerate(trajectories):
        for point_idx, point in enumerate(trajectory):
            px, py = point[0], point[1]
            dist = math.hypot(px - x, py - y)
            if dist <= SELECT_RADIUS and dist < min_dist:
                min_dist = dist
                closest_traj_idx = traj_idx
    
    if closest_traj_idx != -1:
        current_traj_idx = closest_traj_idx
        updated_img = draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point)
        return updated_img, trajectories, current_traj_idx, temp_point
    
    if current_traj_idx == -1 or current_traj_idx >= len(trajectories) or len(trajectories[current_traj_idx]) == 0:
        if current_traj_idx == -1 or current_traj_idx >= len(trajectories):
            trajectories.append([])
            current_traj_idx = len(trajectories) - 1
        
        trajectories[current_traj_idx].append([x, y])
        trajectories.append([])
        next_idx = len(trajectories) - 1
        current_traj_idx = next_idx
        
    elif len(trajectories[current_traj_idx]) == 1:
        trajectories[current_traj_idx].append([x, y])
        trajectories.append([])
        next_idx = len(trajectories) - 1
        current_traj_idx = next_idx
        
    else:
        trajectories[current_traj_idx].append([x, y])

    updated_img = draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point)
    return updated_img, trajectories, current_traj_idx, temp_point

# ==================== 按钮功能 ====================
def clear_all_trajectories(original_image):
    if original_image is None:
        return None, [], -1, None
    updated_img = draw_trajectories_on_image(original_image, [], -1, None)
    return updated_img, [], -1, None

def create_new_trajectory_group(original_image, trajectories, current_traj_idx, temp_point):
    if original_image is None:
        return original_image, trajectories, -1, temp_point
    current_traj_idx = len(trajectories)
    trajectories.append([])
    updated_img = draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point)
    return updated_img, trajectories, current_traj_idx, temp_point

def undo_last_point_from_current_trajectory(original_image, trajectories, current_traj_idx, temp_point):
    if original_image is None or current_traj_idx < 0 or current_traj_idx >= len(trajectories):
        return draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point), trajectories, current_traj_idx, temp_point
    current_trajectory = trajectories[current_traj_idx]
    if len(current_trajectory) > 0:
        current_trajectory.pop()
        print(f"✅ 已撤销轨迹组 {current_traj_idx} 的最后一个点。")
    else:
        print(f"⚠️ 轨迹组 {current_traj_idx} 为空，无法撤销。")
    updated_img = draw_trajectories_on_image(original_image, trajectories, current_traj_idx, temp_point)
    return updated_img, trajectories, current_traj_idx, temp_point

def generate_output_new(trajectories_list, original_image_state, use_wan,num_frames):
    if original_image_state is None:
        H, W = 512, 512
        tracks = torch.zeros(1, num_frames, 0, 2)
        path = visualize(tracks, (W, H))
        return path
    
    tracks_tensor = convert_trajectories_to_tensor(trajectories_list, num_frames)  
    path = visualize(tracks_tensor, original_image_state.size, temp=not use_wan)
    
    if use_wan:
        print("🎬 开始生成 Live2Diff 视频...")
        path = wan_flow(original_image_state, path)
    else:
        print("⏭️ 跳过 Live2Diff，仅输出轨迹可视化。")
        
    return path

# ==================== Gradio 界面（中文版） ====================
with gr.Blocks() as demo:
    gr.Markdown("## 🖱️ Live2Diff：轨迹标注工具（点击版）")
    gr.Markdown("帧数格式：1+4K → 仅允许 5、9、13、17")
    
    original_image_state = gr.State(None)
    trajectories = gr.State([])
    current_traj_idx = gr.State(-1)
    temp_point = gr.State(None)

    with gr.Row():
        with gr.Column():
            img_display = gr.Image(label="上传图片并在此标注轨迹", interactive=True, type="pil")
            
            frame_selector = gr.Radio(
                choices=[5, 9, 13, 17],
                value=9,
                label="选择总帧数（1+4K格式）",
                interactive=True
            )
            
            wan_checkbox = gr.Checkbox(label="启用 Live2Diff 视频生成（速度较慢）", value=False)
            
            with gr.Row():
                clear_btn = gr.Button("清空全部")
                new_traj_btn = gr.Button("新建轨迹")
                undo_traj_btn = gr.Button("撤销上一点")
            
            generate_btn = gr.Button("生成视频", variant="primary", size="lg")
        
        with gr.Column():
            output_video = gr.Video(label="输出视频")

    def on_upload(img: Image.Image):
        if img is None:
            return None, None, [], -1, None
        displayed = draw_trajectories_on_image(img, [], -1, None)
        return img, displayed, [], -1, None

    img_display.upload(
        fn=on_upload,
        inputs=img_display,
        outputs=[original_image_state, img_display, trajectories, current_traj_idx, temp_point]
    )

    img_display.select(
        fn=handle_click_new,
        inputs=[original_image_state, trajectories, current_traj_idx, temp_point],
        outputs=[img_display, trajectories, current_traj_idx, temp_point],
        show_progress="hidden"
    )

    new_traj_btn.click(
        fn=create_new_trajectory_group,
        inputs=[original_image_state, trajectories, current_traj_idx, temp_point],
        outputs=[img_display, trajectories, current_traj_idx, temp_point]
    )

    undo_traj_btn.click(
        fn=undo_last_point_from_current_trajectory,
        inputs=[original_image_state, trajectories, current_traj_idx, temp_point],
        outputs=[img_display, trajectories, current_traj_idx, temp_point]
    )

    clear_btn.click(
        fn=clear_all_trajectories,
        inputs=[original_image_state],
        outputs=[img_display, trajectories, current_traj_idx, temp_point]
    )

    generate_btn.click(
        fn=generate_output_new,
        inputs=[trajectories, original_image_state, wan_checkbox, frame_selector],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch()