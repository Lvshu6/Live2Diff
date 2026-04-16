import os
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw
import flow_vis


class DirectionVisualizer:
    def __init__(
        self,
        save_dir="./results",
        linewidth=4,
        fps=10,
        history=3,
        save_pure_trajectory=True,  # 新增选项：是否保存纯轨迹视频
    ):
        self.save_dir = save_dir
        self.linewidth = linewidth
        self.fps = fps
        self.history = history
        self.save_pure_trajectory = save_pure_trajectory

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        filename="direction_tracks",
        save_video=True,
    ):
        video = video[0].permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3)
        tracks = tracks[0].cpu().numpy().astype(np.float32)  # (T, N, 2)

        T, H, W, _ = video.shape
        N = tracks.shape[1]

        overlay_frames = []   # 原视频 + 轨迹
        pure_frames = []      # 纯黑背景 + 轨迹（仅当 save_pure_trajectory=True）

        # # 预先创建纯黑背景（如果需要）
        # if self.save_pure_trajectory:
        #     black_bg = np.zeros((H, W, 3), dtype=np.uint8)
        # 预先创建纯白背景（修改：从纯黑改为纯白）
        if self.save_pure_trajectory:
            black_bg = np.ones((H, W, 3), dtype=np.uint8) * 255  # 纯白背景

        for t in range(T):
            # --- 原视频帧 ---
            overlay_frame = Image.fromarray(video[t].astype(np.uint8))
            overlay_draw = ImageDraw.Draw(overlay_frame)

            # --- 纯轨迹帧（可选）---
            if self.save_pure_trajectory:
                pure_frame = Image.fromarray(black_bg.copy())
                pure_draw = ImageDraw.Draw(pure_frame)
            else:
                pure_draw = None

            # ----------- 绘制历史方向线（使用 flow_vis 风格）-----------
            for k in range(1, self.history + 1):
                if t - k < 0:
                    continue

                prev = tracks[t - k]      # (N, 2)
                curr = tracks[t - k + 1]  # (N, 2)

                # 计算动态线宽
                min_width = 2
                max_width = self.linewidth
                if self.history > 1:
                    current_width = int(
                        max_width - (k - 1) * (max_width - min_width) / (self.history - 1)
                    )
                else:
                    current_width = max_width
                current_width = max(min_width, current_width)

                for i in range(N):
                    x1, y1 = prev[i]
                    x2, y2 = curr[i]

                    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                        continue

                    # 使用 flow_vis 着色
                    dx = x2 - x1
                    dy = y2 - y1
                    flow_vec = np.array([[[dx, dy]]], dtype=np.float32)
                    color_img = flow_vis.flow_to_color(flow_vec)
                    color = tuple(color_img[0, 0].tolist())

                    # alpha_scale = (self.history - k + 1) / self.history
                    
                    # faded_color = tuple(int(c * alpha_scale) for c in color)

                    # 绘制到原视频
                    # 直接使用 flow_vis 的颜色，不做透明度衰减
                    overlay_draw.line((x1, y1, x2, y2), fill=color, width=current_width)

                    if self.save_pure_trajectory:
                        pure_draw.line((x1, y1, x2, y2), fill=color, width=current_width)


            overlay_frames.append(np.array(overlay_frame))
            if self.save_pure_trajectory:
                pure_frames.append(np.array(pure_frame))

        result_tensor = torch.from_numpy(
            np.stack(pure_frames)
        ).permute(0, 3, 1, 2)[None].byte()

        # 保存视频
        if save_video:
            self._save_video(overlay_frames, filename+ "_pure")
            if self.save_pure_trajectory:
                self._save_video(pure_frames, filename)
        pure_frames_pil = [Image.fromarray(frame) for frame in pure_frames]
        return pure_frames_pil

    def _draw_circle_on_image(self, img, coord, radius, color):
        """辅助函数：在 PIL Image 上画圆"""
        draw = ImageDraw.Draw(img)
        x, y = int(coord[0]), int(coord[1])
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], fill=color, outline=color)

    def _save_video(self, frames, filename):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        writer = imageio.get_writer(save_path, fps=self.fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Saved to {save_path}")