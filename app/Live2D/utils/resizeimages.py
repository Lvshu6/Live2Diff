import os
from PIL import Image

# ====================== 配置区（只改这里） ======================
# 你的【总文件夹】路径（里面有很多子文件夹装图片）
INPUT_ROOT = r"/home/suat/yxd/DiffSynth-Studionew/webui3/pet/images"
# 输出【总文件夹】路径（自动复刻子文件夹结构）
OUTPUT_ROOT = r"/home/suat/yxd/DiffSynth-Studionew/webui3/pet/images_resized"
# 目标缩放尺寸（宽, 高）
TARGET_SIZE = (480, 848)
# =================================================================

# 支持的图片格式
IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

def resize_all_images():
    # 遍历所有子文件夹 + 文件
    for root_dir, sub_dirs, files in os.walk(INPUT_ROOT):
        # 构建输出文件夹路径（保持原目录结构）
        relative_path = os.path.relpath(root_dir, INPUT_ROOT)
        output_dir = os.path.join(OUTPUT_ROOT, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # 遍历当前文件夹里的所有文件
        for file in files:
            # 只处理图片
            if file.lower().endswith(IMAGE_FORMATS):
                input_img_path = os.path.join(root_dir, file)
                output_img_path = os.path.join(output_dir, file)

                try:
                    # 打开 + 高质量缩放
                    with Image.open(input_img_path) as img:
                        # LANCZOS 最好的缩放质量
                        resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        resized_img.save(output_img_path, quality=95)
                    
                    print(f"✅ 成功：{input_img_path}")

                except Exception as e:
                    print(f"❌ 失败：{input_img_path} | 错误：{str(e)}")

if __name__ == "__main__":
    print("开始递归缩放所有子文件夹图片...\n")
    resize_all_images()
    print("\n✅ 全部处理完成！")