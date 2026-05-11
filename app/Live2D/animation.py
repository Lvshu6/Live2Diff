"""
动画状态机 —— 接收 (dx,dy) 偏移序列，在图谱中寻路导航，驱动图像切换。
纯逻辑层，零 LLM 依赖，零 Qt 依赖。
"""

import random

import networkx as nx
import models
from utils import calculate_loss


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
        # self._idle_count = 0

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def idle_enabled(self) -> bool:
        return self._idle_enabled

    def set_idle_enabled(self, enabled: bool):
        self._idle_enabled = enabled
        self._idle_waiting = False
        print(f"[待机] 待机动画 {'开启' if enabled else '关闭'}")

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
        # self._idle_count = 0

        center = self._callbacks["get_image_center"]()
        self._initial_img_center = center

        print(
            f"[动画] 开始: origin={self._origin_node}, img_center={self._initial_img_center}"
        )
        for i, (dx, dy) in enumerate(offsets, 1):
            print(f"[动画]   motion {i}: dx={dx:.2f}, dy={dy:.2f}")

        self._callbacks["on_status"](f"执行动画... {describe}")
        self._process_next_motion()

    def tick(self):
        if self._is_active and self._waiting_for_queue:
            if self._callbacks["is_queue_empty"]():
                self._waiting_for_queue = False
                print(
                    f"[检查] 队列已空, step={self._motion_step}, returning={self._returning_to_center}"
                )

                if self._returning_to_center:
                    current = self._callbacks["get_node_path"]()
                    if current == self._origin_node:
                        print("[检查] 回中完成，已到达起点")
                        self._finish_animation()
                    else:
                        print(
                            f"[检查] 回中未完成，当前={current}，目标={self._origin_node}"
                        )
                        self._waiting_for_queue = True
                        self._continue_return_path()
                    return

                current = self._callbacks["get_node_path"]()
                if self._motion_target_node and current != self._motion_target_node:
                    print(
                        f"[检查] 未到达目标 {self._motion_target_node}，当前={current}，继续填充路径"
                    )
                    self._waiting_for_queue = True
                    self._continue_motion_path()
                    return

                self._motion_step += 1
                print(f"[检查] 立即执行下一步 (step={self._motion_step})")

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
            self._do_idle_step()

    def navigate_offset(self, dx, dy):
        track = self._callbacks["get_track_points"]()
        if not track:
            return
        t1_target = [(x + dx, y + dy) for x, y in track]
        self._idle_enabled = False
        self._find_and_add_path(t1_target)

    def _process_next_motion(self):
        print(
            f"[动画] _process_next_motion step={self._motion_step}/{len(self._motion_sequence)}"
        )
        if self._motion_step >= len(self._motion_sequence):
            print("[动画] 所有 motion 完成，准备回中")
            self._start_return_to_center()
            return

        cx, cy = self._motion_sequence[self._motion_step]
        cy = -cy

        track = self._callbacks["get_track_points"]()
        if not track:
            print(f"[动画] track 为空，跳过 motion {self._motion_step}")
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
        print(
            f"[动画] motion {step_label}: img_center=({img_cx:.0f},{img_cy:.0f}), llm_offset=({cx:.1f},{cy:.1f}), target=({target_cx:.1f},{target_cy:.1f}), face=({face_cx:.1f},{face_cy:.1f}), delta=({dx:.1f},{dy:.1f})"
        )
        self._callbacks["on_status"](
            f"动作 {step_label}/{len(self._motion_sequence)}: target->({target_cx:.0f},{target_cy:.0f})  {self._describe}"
        )

        self._find_and_add_path(t1_target)
        self._waiting_for_queue = True

    def _find_and_add_path(self, t1_target):
        start_node = self._callbacks["get_node_path"]()
        print(f"[寻路] start_node={start_node}, t1_target样本={t1_target[:2]}")

        if models.G is None or start_node not in models.G:
            print(
                f"[寻路] 起点不在图中: models.G={models.G is None}, in_G={start_node in models.G if models.G else 'N/A'}"
            )
            return

        node_loss = {}
        count = 0
        for node in models.G.nodes:
            node_pos = models.NODE_POSITIONS.get(node, [])
            loss = calculate_loss(t1_target, node_pos)
            node_loss[node] = loss
            count += 1

        print(f"[寻路] 全局搜索了 {count} 个节点")

        if not node_loss:
            print("[寻路] 没有可用的节点")
            return

        target_node = min(node_loss, key=node_loss.get)
        self._motion_target_node = target_node
        min_loss = node_loss[target_node]
        best_pos = models.NODE_POSITIONS.get(target_node, [])
        print(
            f"[寻路] 目标节点={target_node}, min_loss={min_loss:.4f}, best_pos样本={best_pos[:2]}"
        )
        if min_loss > 10000:
            print(f"[寻路] WARNING: min_loss={min_loss:.1f} 很大，可能未找到匹配节点")

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
            print(f"[寻路] 最短路径长度={len(shortest)}: {shortest[:3]}...")
        except nx.NetworkXNoPath:
            print("[寻路] 无路径，使用起点自身")
            shortest = [start_node]

        added = self._callbacks["add_paths_to_queue"](shortest[1:])
        print(f"[寻路] 队列添加了 {added} 个节点 (目标={target_node})")

    def _start_return_to_center(self):
        print(f"[回中] 开始回中: origin={self._origin_node}")
        if not self._origin_node:
            print("[回中] 无 origin 节点")
            self._finish_animation()
            return

        current = self._callbacks["get_node_path"]()
        print(f"[回中] current={current}")
        if current == self._origin_node:
            print("[回中] 已在起点")
            self._finish_animation()
            return

        if (
            models.G is None
            or current not in models.G
            or self._origin_node not in models.G
        ):
            print(f"[回中] 节点不在图中: G={models.G is None}")
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, self._origin_node)
            print(f"[回中] 路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print("[回中] 无路径")
            self._finish_animation()
            return

        added = self._callbacks["add_paths_to_queue"](path[1:])
        print(f"[回中] 队列添加了 {added} 个节点")

        self._returning_to_center = True
        self._waiting_for_queue = True
        self._callbacks["on_status"]("回中...")

    def _finish_animation(self):
        print(f"[动画] _finish_animation: describe='{self._describe}'")
        self._is_active = False
        self._waiting_for_queue = False
        self._returning_to_center = False
        self._motion_target_node = None
        self._callbacks["on_complete"]()

    def _continue_motion_path(self):
        current = self._callbacks["get_node_path"]()
        target = self._motion_target_node
        print(f"[续路] current={current}, target={target}")

        if (
            not target
            or not models.G
            or current not in models.G
            or target not in models.G
        ):
            print(f"[续路] 无法继续寻路，跳过当前 motion")
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
            print(f"[续路] 补充路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print(f"[续路] 无路径到目标，跳过当前 motion")
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        added = self._callbacks["add_paths_to_queue"](path[1:])
        print(f"[续路] 补充了 {added} 个节点")
        if added == 0:
            self._waiting_for_queue = True

    def _continue_return_path(self):
        current = self._callbacks["get_node_path"]()
        target = self._origin_node
        print(f"[续回] current={current}, target={target}")

        if (
            not target
            or not models.G
            or current not in models.G
            or target not in models.G
        ):
            print(f"[续回] 无法继续回中，直接结束")
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
            print(f"[续回] 补充路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print(f"[续回] 无路径到起点，直接结束")
            self._finish_animation()
            return

        added = self._callbacks["add_paths_to_queue"](path[1:])
        print(f"[续回] 补充了 {added} 个节点")
        if added == 0:
            self._waiting_for_queue = True

    def _do_idle_step(self):
        track = self._callbacks["get_track_points"]()
        if not track:
            return

        # if self._idle_count != 3:
        #     self._idle_count += 1
        #     return
        # else:
        #     self._idle_count = 0


        dx = random.uniform(-5, 5)
        dy = random.uniform(-20, 20)
        t1_target = [(x + dx, y + dy) for x, y in track]

        start_node = self._callbacks["get_node_path"]()
        print(f"[待机] dx={dx:.1f}, dy={dy:.1f}, start={start_node}")

        if models.G is None or start_node not in models.G:
            return

        node_loss = {}
        for node in models.G.nodes:
            node_pos = models.NODE_POSITIONS.get(node, [])
            loss = calculate_loss(t1_target, node_pos)
            node_loss[node] = loss

        if not node_loss:
            return

        target_node = min(node_loss, key=node_loss.get)
        if target_node == start_node:
            return

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
        except nx.NetworkXNoPath:
            return

        self._callbacks["add_paths_to_queue"](shortest[1:])

        self._start_return_to_center()

        print(f"[待机] 路径长度={len(shortest)} → {target_node}")
