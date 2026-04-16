import os
import glob
import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

# ====================== 配置项 ======================
SOURCE_VIDEO_DIR = "/home/suat/yxd/DiffSynth-Studionew/app/Live2D/pet/videos"
TARGET_VIDEO_DIR = "/home/suat/yxd/DiffSynth-Studionew/app/Live2D/pet/ref"
OUTPUT_VIDEO_DIR = SOURCE_VIDEO_DIR + "/resized_videos"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
# ==================================================

def get_video_dimensions(video_path):
    """获取视频宽高"""
    try:
        # 读取第一帧
        first_frame = iio.imread(video_path, index=0)
        if len(first_frame.shape) >= 2:
            h, w = first_frame.shape[:2]
            return w, h
        return None, None
    except Exception as e:
        print(f"⚠️ 无法获取尺寸 {video_path}: {e}")
        return None, None

def get_video_fps(video_path):
    """获取视频帧率"""
    try:
        props = iio.improps(video_path)
        # imageio v3 props 对象通常有 fps 属性
        return getattr(props, 'fps', 30.0)
    except:
        return 30.0

def resize_frame_pil(frame, target_w, target_h):
    """
    使用 PIL 进行高质量 Resize
    frame: numpy array (H, W, C) or (H, W)
    returns: numpy array
    """
    # 转 PIL Image
    img = Image.fromarray(frame)
    
    # Resize (注意 PIL 参数顺序是 (width, height))
    # LANCZOS 是高质量下采样滤波器，适合视频缩放
    resized_img = img.resize((target_w, target_h), Image.LANCZOS)
    
    # 转回 numpy
    return np.array(resized_img)

def process_single_video(source_path, target_w, target_h, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fps = get_video_fps(source_path)
        print(f"📥 读取: {os.path.basename(source_path)} (目标: {target_w}x{target_h})")

        # 【关键】一次性读取所有帧，避免循环 index 导致的兼容性问题
        # index=None 告诉 imageio 读取整个视频序列
        frames = iio.imread(source_path, index=None)
        
        if frames.ndim == 3:
            # 如果是灰度图或单帧被读成 (H, W, C)，增加时间维度 (1, H, W, C)
            frames = frames[np.newaxis, ...]
            
        total_frames = frames.shape[0]
        resized_frames = []

        # 逐帧处理
        for i in tqdm(range(total_frames), desc="Resizing (PIL)", leave=False):
            frame = frames[i]
            resized_frame = resize_frame_pil(frame, target_w, target_h)
            resized_frames.append(resized_frame)
        
        # 保存视频
        print(f"📤 保存: {os.path.basename(output_path)}")
        iio.imwrite(
            output_path,
            resized_frames,
            fps=fps,
            codec="h264" # 如果环境不支持 h264，可尝试移除该参数或使用 "libx264"
        )
        
        print(f"✅ 完成: {output_path}")
        return True

    except Exception as e:
        print(f"❌ 失败 {source_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_resize_videos():
    # 1. 构建参考映射
    target_map = {}
    print("🔍 扫描参考视频文件夹...")
    for path in glob.glob(os.path.join(TARGET_VIDEO_DIR, "*")):
        if path.lower().endswith(VIDEO_EXTENSIONS):
            name = os.path.splitext(os.path.basename(path))[0]
            w, h = get_video_dimensions(path)
            if w and h:
                target_map[name] = (w, h)
    
    if not target_map:
        print("❌ 参考文件夹未找到有效视频")
        return

    # 2. 处理源视频
    print("🚀 开始批量处理...")
    for path in glob.glob(os.path.join(SOURCE_VIDEO_DIR, "*")):
        if path.lower().endswith(VIDEO_EXTENSIONS):
            name = os.path.splitext(os.path.basename(path))[0]
            
            if name not in target_map:
                print(f"⏭️ 跳过 (无参考): {name}")
                continue
            
            out_path = os.path.join(OUTPUT_VIDEO_DIR, f"{name}.mp4")
            if os.path.exists(out_path):
                print(f"⏭️ 跳过 (已存在): {name}")
                continue
            
            tw, th = target_map[name]
            process_single_video(path, tw, th, out_path)

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    batch_resize_videos()