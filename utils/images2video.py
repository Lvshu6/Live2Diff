import os
import imageio.v3 as iio
from pathlib import Path

# 设置你的根目录（包含多个图片文件夹）
root_dir = Path(r"D:\图片\track视频")  # 修改为你自己的路径
output_dir = Path("output")  # 或者设为另一个路径，如 Path("path/to/output_videos")

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 支持的图像扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}

for folder in root_dir.iterdir():
    if not folder.is_dir():
        continue  # 跳过非文件夹项

    # 获取该文件夹中所有图像文件，并排序（确保顺序一致）
    image_files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if not image_files:
        print(f"跳过空文件夹: {folder.name}")
        continue

    # 构造输出视频路径：与文件夹同名，扩展名为 .mp4
    video_path = output_dir / f"{folder.name}.mp4"

    # 读取所有图像
    images = [iio.imread(f) for f in image_files]

    # 写入视频（默认帧率 30 fps，可调整）
    iio.imwrite(video_path, images, fps=5)

    print(f"已生成视频: {video_path}")