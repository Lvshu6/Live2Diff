"""
动画状态机 —— 接收 (dx,dy) 偏移序列，在图谱中寻路导航，驱动图像切换。
纯逻辑层，零 LLM 依赖，零 Qt 依赖。
"""

import random

import networkx as nx
import models
from utils import calculate_loss

DEPTH_LIMIT = 10
QUEUE_LEN = 10
MAX_QUEUE_ADD = 10
class AnimationController:
    def __init__(self, callbacks: dict):
        self._callbacks = callbacks

        self._is_active = False
        self._waiting_for_queue = False
        self._motion_sequence = []
        self._motion_step = 0
        self._origin_node = None
        self._returning_to_center = False
        self._describe = ""
        self._motion_target_node = None
        self._initial_img_center = None

        self._idle_enabled = False
        self._idle_waiting = False
        self._idle_cooldown = 0
        self._idle_cooldown_ticks = 20   # ~1.6 s pause between idle steps

        self._return_to_center_enabled = True

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def idle_enabled(self) -> bool:
        return self._idle_enabled

    def set_idle_enabled(self, enabled: bool):
        self._idle_enabled = enabled
        self._idle_waiting = False
        self._idle_cooldown = 0

    def set_return_to_center(self, enabled: bool):
        self._return_to_center_enabled = enabled

    def start_by_path(self, paths: list, describe: str = ""):
        if self._is_active or not paths:
            return

        self._is_active = True
        self._motion_sequence = []
        self._motion_step = 0
        self._origin_node = self._callbacks["get_node_path"]()
        self._returning_to_center = False
        self._waiting_for_queue = False
        self._describe = describe
        self._motion_target_node = paths[-1]
        self._idle_waiting = False
        self._initial_img_center = self._callbacks["get_image_center"]()

        self._callbacks["on_status"](f"执行动画... {describe}")
        self._callbacks["add_paths_to_queue"](paths)
        self._waiting_for_queue = True

    def start(self, offsets, describe=""):
        if self._is_active:
            return

        track = self._callbacks["get_track_points"]()
        if not track:
            return

        self._is_active = True
        self._motion_sequence = offsets
        self._motion_step = 0
        self._origin_node = self._callbacks["get_node_path"]()
        self._returning_to_center = False
        self._waiting_for_queue = False
        self._describe = describe
        self._motion_target_node = None
        self._idle_waiting = False

        self._initial_img_center = self._callbacks["get_image_center"]()
        self._callbacks["on_status"](f"执行动画... {describe}")
        self._process_next_motion()

    def tick(self):
        if self._is_active and self._waiting_for_queue:
            if self._callbacks["is_queue_empty"]():
                self._waiting_for_queue = False

                if self._returning_to_center:
                    current = self._callbacks["get_node_path"]()
                    if current == self._origin_node:
                        self._finish_animation()
                    else:
                        self._waiting_for_queue = True
                        self._continue_return_path()
                    return

                current = self._callbacks["get_node_path"]()
                if self._motion_target_node and current != self._motion_target_node:
                    self._waiting_for_queue = True
                    self._continue_motion_path()
                    return

                self._motion_step += 1
                if self._motion_step >= len(self._motion_sequence):
                    self._callbacks["on_status"]("动作完成，回中...")
                    self._start_return_to_center()
                else:
                    self._process_next_motion()

        if (
            self._idle_enabled
            and not self._is_active
            and self._callbacks["is_queue_empty"]()
        ):
            if self._idle_cooldown > 0:
                self._idle_cooldown -= 1
            else:
                self._do_idle_step()

    def navigate_offset(self, dx, dy):
        track = self._callbacks["get_track_points"]()
        if not track:
            return
        t1_target = [(x + dx, y + dy) for x, y in track]
        self._idle_enabled = False
        self._find_and_add_path(t1_target)

    def _process_next_motion(self):
        if self._motion_step >= len(self._motion_sequence):
            self._start_return_to_center()
            return

        cx, cy = self._motion_sequence[self._motion_step]
        cy = -cy

        track = self._callbacks["get_track_points"]()
        if not track:
            self._motion_step += 1
            self._process_next_motion()
            return

        if self._initial_img_center:
            img_cx, img_cy = self._initial_img_center
        else:
            center = self._callbacks["get_image_center"]()
            img_cx = center[0] if center else 0
            img_cy = center[1] if center else 0

        target_cx = img_cx + cx
        target_cy = img_cy + cy
        face_cx = sum(p[0] for p in track) / len(track)
        face_cy = sum(p[1] for p in track) / len(track)
        dx = target_cx - face_cx
        dy = target_cy - face_cy
        t1_target = [(x + dx, y + dy) for x, y in track]

        step_label = self._motion_step + 1
        self._callbacks["on_status"](
            f"动作 {step_label}/{len(self._motion_sequence)}: {self._describe}"
        )
        self._find_and_add_path(t1_target)
        self._waiting_for_queue = True

    def _find_and_add_path(self, t1_target):
        start_node = self._callbacks["get_node_path"]()
        if models.G is None or start_node not in models.G:
            return

        node_loss = {
            node: calculate_loss(t1_target, models.NODE_POSITIONS.get(node, []))
            for node in models.G.nodes
        }
        if not node_loss:
            return

        target_node = min(node_loss, key=node_loss.get)
        self._motion_target_node = target_node

        if node_loss[target_node] > 10000:
            print(f"[animation] WARNING: min_loss 很大，可能未找到匹配节点")

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
        except nx.NetworkXNoPath:
            shortest = [start_node]

        self._callbacks["add_paths_to_queue"](shortest[1:])

    def _start_return_to_center(self):
        # idle 动画始终回中；只有语义动作（describe != "idle"）受开关控制
        if not self._return_to_center_enabled and self._describe != "idle":
            self._finish_animation()
            return

        if not self._origin_node:
            self._finish_animation()
            return

        current = self._callbacks["get_node_path"]()
        if current == self._origin_node:
            self._finish_animation()
            return

        if (
            models.G is None
            or current not in models.G
            or self._origin_node not in models.G
        ):
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, self._origin_node)
        except nx.NetworkXNoPath:
            self._finish_animation()
            return

        self._callbacks["add_paths_to_queue"](path[1:])
        self._returning_to_center = True
        self._waiting_for_queue = True
        self._callbacks["on_status"]("回中...")

    def _finish_animation(self):
        self._is_active = False
        self._waiting_for_queue = False
        self._returning_to_center = False
        self._motion_target_node = None
        self._idle_cooldown = self._idle_cooldown_ticks
        self._callbacks["on_complete"]()

    def _continue_motion_path(self):
        current = self._callbacks["get_node_path"]()
        target = self._motion_target_node

        if not target or not models.G or current not in models.G or target not in models.G:
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
        except nx.NetworkXNoPath:
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        added = self._callbacks["add_paths_to_queue"](path[1:])
        if added == 0:
            self._waiting_for_queue = True

    def _continue_return_path(self):
        current = self._callbacks["get_node_path"]()
        target = self._origin_node

        if not target or not models.G or current not in models.G or target not in models.G:
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
        except nx.NetworkXNoPath:
            self._finish_animation()
            return

        added = self._callbacks["add_paths_to_queue"](path[1:])
        if added == 0:
            self._waiting_for_queue = True

    def _do_idle_step(self):
        track = self._callbacks["get_track_points"]()
        if not track:
            return

        dx = random.uniform(-20, 20)
        dy = random.uniform(-20, 20)
        t1_target = [(x + dx, y + dy) for x, y in track]
        start_node = self._callbacks["get_node_path"]()

        if models.G is None or start_node not in models.G:
            return

        node_loss = {
            node: calculate_loss(t1_target, models.NODE_POSITIONS.get(node, []))
            for node in models.G.nodes
        }
        if not node_loss:
            return

        target_node = min(node_loss, key=node_loss.get)
        if target_node == start_node:
            return

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
        except nx.NetworkXNoPath:
            return

        self._is_active = True
        self._origin_node = start_node
        self._motion_sequence = []
        self._motion_step = 0
        self._motion_target_node = target_node
        self._returning_to_center = False
        self._waiting_for_queue = True
        self._describe = "idle"

        self._callbacks["add_paths_to_queue"](shortest[1:])
