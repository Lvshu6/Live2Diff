import os
from pathlib import Path
import imageio.v3 as iio

# 设置你的视频根目录（包含多个视频文件）
video_dir = Path(r"/home/suat/yxd/DiffSynth-Studionew/gradio_output/1")  # 修改为你自己的路径
output_root = Path("/home/suat/yxd/DiffSynth-Studionew/gradio_images")   # 输出的根目录

# 确保输出根目录存在
output_root.mkdir(parents=True, exist_ok=True)

# 支持的视频扩展名（根据 imageio 支持的格式）
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

for video_path in video_dir.iterdir():
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        continue

    # 创建对应输出文件夹：output_root / video_name_without_ext
    folder_name = video_path.stem  # 不含扩展名的文件名
    output_folder = output_root / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"正在处理视频: {video_path.name} → 保存到 {output_folder}")

    # 读取视频并逐帧保存
    try:
        reader = iio.imiter(video_path, plugin="pyav")  # 使用 pyav 插件更稳定
        for idx, frame in enumerate(reader):
            img_path = output_folder / f"{idx:06d}.png"  # 000000.png, 000001.png, ...
            iio.imwrite(img_path, frame)
    except Exception as e:
        print(f"⚠️ 处理 {video_path.name} 时出错: {e}")
        continue

    print(f"✅ 完成: {video_path.name} 共提取 {idx+1} 帧")