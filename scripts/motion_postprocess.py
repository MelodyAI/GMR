#!/usr/bin/env python3
"""Quality diagnostics and preview-oriented post-processing for GMR motions.

The supported input is the standard GMR robot_motion.pkl schema:
fps/root_pos/root_rot/dof_pos/local_body_pos/link_body_list. The optimizer is
designed for preview cleanup and data diagnosis; it is not a real-robot safety
filter.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

GMR_ROOT = Path(__file__).resolve().parents[1]
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

import mujoco as mj  # noqa: E402

from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402
# from gmr_web.runner import render_robot_motion  # noqa: E402


@dataclass(frozen=True)
class JointInfo:
    name: str
    dof_index: int
    qpos_index: int
    qrange: tuple[float, float]
    velocity_limit: float
    accel_limit: float
    jerk_limit: float
    smooth_window: int


@dataclass(frozen=True)
class Profile:
    waist_velocity: float
    arm_velocity: float
    wrist_velocity: float
    leg_velocity: float
    default_velocity: float
    waist_accel: float
    arm_accel: float
    wrist_accel: float
    leg_accel: float
    default_accel: float
    waist_jerk: float
    arm_jerk: float
    wrist_jerk: float
    leg_jerk: float
    default_jerk: float
    arm_window: int
    leg_window: int
    waist_window: int
    wrist_window: int
    default_window: int
    root_window: int


# 这些 profile 只是预览后处理的参数预设，不是真机安全限制。
# preview/soft/strict 的区别主要是“去抖和平滑”的力度逐渐增强。
PROFILES = {
    "preview": Profile(
        waist_velocity=4.5,
        arm_velocity=8.0,
        wrist_velocity=10.0,
        leg_velocity=7.0,
        default_velocity=8.0,
        waist_accel=80.0,
        arm_accel=120.0,
        wrist_accel=160.0,
        leg_accel=110.0,
        default_accel=120.0,
        waist_jerk=1600.0,
        arm_jerk=2400.0,
        wrist_jerk=3200.0,
        leg_jerk=2200.0,
        default_jerk=2400.0,
        arm_window=11,
        leg_window=9,
        waist_window=7,
        wrist_window=9,
        default_window=7,
        root_window=7,
    ),
    "soft": Profile(
        waist_velocity=3.5,
        arm_velocity=5.5,
        wrist_velocity=7.0,
        leg_velocity=5.5,
        default_velocity=6.0,
        waist_accel=55.0,
        arm_accel=75.0,
        wrist_accel=100.0,
        leg_accel=80.0,
        default_accel=85.0,
        waist_jerk=900.0,
        arm_jerk=1400.0,
        wrist_jerk=1900.0,
        leg_jerk=1500.0,
        default_jerk=1500.0,
        arm_window=17,
        leg_window=13,
        waist_window=11,
        wrist_window=13,
        default_window=11,
        root_window=11,
    ),
    "strict": Profile(
        waist_velocity=2.5,
        arm_velocity=4.0,
        wrist_velocity=5.0,
        leg_velocity=4.0,
        default_velocity=4.5,
        waist_accel=35.0,
        arm_accel=50.0,
        wrist_accel=70.0,
        leg_accel=55.0,
        default_accel=60.0,
        waist_jerk=600.0,
        arm_jerk=900.0,
        wrist_jerk=1200.0,
        leg_jerk=1000.0,
        default_jerk=1000.0,
        arm_window=21,
        leg_window=17,
        waist_window=15,
        wrist_window=17,
        default_window=15,
        root_window=15,
    ),
}


# 脚底指标尽量兼容不同机器人模型：如果模型里没有匹配到脚/脚踝相关 body，
# 报告里会标记 contact unavailable，而不是让整个任务失败。
FOOT_NAME_HINTS = ("foot", "toe", "sole", "ankle")
ARM_NAME_HINTS = ("shoulder", "elbow", "wrist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose and post-process standard GMR robot_motion.pkl files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality = subparsers.add_parser("quality", help="Only write a quality report.")
    add_common_args(quality)
    quality.add_argument(
        "--output",
        default=None,
        help="Quality JSON path. Defaults to motion_quality.json next to input.",
    )

    optimize = subparsers.add_parser("optimize", help="Write an optimized pkl and quality report.")
    add_common_args(optimize)
    optimize.add_argument(
        "--profile",
        default="preview",
        choices=sorted(PROFILES),
        help="Optimization profile.",
    )
    optimize.add_argument(
        "--pipeline",
        default="v2",
        choices=["legacy", "v2", "v2_foot"],
        help="Optimization pipeline. legacy keeps the old full smoothing behavior; v2 focuses on arm jitter cleanup; v2_foot adds light foot-lock cleanup.",
    )
    optimize.add_argument(
        "--output",
        default=None,
        help="Optimized pkl path. Defaults to short names such as motion_soft.pkl or motion_foot.pkl.",
    )
    optimize.add_argument(
        "--quality-json",
        default=None,
        help="Quality JSON path. Defaults to short names such as quality_soft.json or quality_foot.json.",
    )
    optimize.add_argument("--render", action="store_true", help="Render optimized mp4.")
    optimize.add_argument(
        "--video-output",
        default=None,
        help="Rendered mp4 path. Defaults to short names such as preview_soft.mp4 or preview_foot.mp4.",
    )
    optimize.add_argument("--width", type=int, default=640)
    optimize.add_argument("--height", type=int, default=480)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input standard GMR robot_motion.pkl")
    parser.add_argument("--robot", default="elf3", help="Robot name registered in GMR params.")


def load_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        motion = pickle.load(f)
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = sorted(required - set(motion))
    if missing:
        raise ValueError(f"Missing required motion fields: {missing}")
    return motion


def save_motion(path: Path, motion: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(motion, f)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_robot_model(robot: str) -> mj.MjModel:
    if robot not in ROBOT_XML_DICT:
        raise ValueError(f"Unknown robot: {robot}")
    return mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))


def parse_motion_arrays(motion: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    fps = int(motion["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape [T, 3], got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape [T, 4], got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must have shape [T, D], got {dof_pos.shape}")
    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError("root_pos, root_rot and dof_pos must have the same frame count.")
    return fps, root_pos, root_rot, dof_pos


def get_joint_info(model: mj.MjModel, dof_count: int, profile: Profile) -> list[JointInfo]:
    # GMR 的 dof_pos 只保存机器人实际关节；MuJoCo qpos 前 7 位是自由根节点，
    # 所以普通关节在 dof_pos 里的下标等于 qpos_index - 7。
    joints: list[JointInfo] = []
    for joint_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_index = int(model.jnt_qposadr[joint_id])
        if name is None or qpos_index < 7:
            continue
        dof_index = qpos_index - 7
        if dof_index < 0 or dof_index >= dof_count:
            continue
        lo, hi = model.jnt_range[joint_id]
        joints.append(
            JointInfo(
                name=name,
                dof_index=dof_index,
                qpos_index=qpos_index,
                qrange=(float(lo), float(hi)),
                velocity_limit=limit_for_joint(name, profile, "velocity"),
                accel_limit=limit_for_joint(name, profile, "accel"),
                jerk_limit=limit_for_joint(name, profile, "jerk"),
                smooth_window=smooth_window_for_joint(name, profile),
            )
        )

    joints.sort(key=lambda item: item.dof_index)
    if len(joints) != dof_count:
        raise ValueError(
            f"Robot exposes {len(joints)} 1-DoF joints, but pkl has {dof_count} dofs."
        )
    return joints


def limit_for_joint(name: str, profile: Profile, limit_type: str) -> float:
    name = name.lower()
    if "waist" in name:
        group = "waist"
    elif "wrist" in name:
        group = "wrist"
    elif "shoulder" in name or "elbow" in name:
        group = "arm"
    elif "hip" in name or "knee" in name or "ankle" in name:
        group = "leg"
    else:
        group = "default"
    return float(getattr(profile, f"{group}_{limit_type}"))


def smooth_window_for_joint(name: str, profile: Profile) -> int:
    name = name.lower()
    if "waist" in name:
        return profile.waist_window
    if "wrist" in name:
        return profile.wrist_window
    if "shoulder" in name or "elbow" in name:
        return profile.arm_window
    if "hip" in name or "knee" in name or "ankle" in name:
        return profile.leg_window
    return profile.default_window


def is_arm_joint(name: str) -> bool:
    name = name.lower()
    return any(hint in name for hint in ARM_NAME_HINTS)


def odd_window(window: int, length: int) -> int:
    if length < 3:
        return 1
    window = max(1, min(int(window), length))
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def triangular_kernel(window: int) -> np.ndarray:
    center = window // 2
    weights = np.asarray([center + 1 - abs(i - center) for i in range(window)], dtype=np.float64)
    return weights / weights.sum()


def centered_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = odd_window(window, len(values))
    if window <= 1:
        return values.copy()

    kernel = triangular_kernel(window)
    pad = window // 2
    if values.ndim == 1:
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed


def velocity_limited_path(values: np.ndarray, max_delta: np.ndarray) -> np.ndarray:
    # 从正向和反向各做一次帧间限速，再取平均。这样既能压住突变，
    # 又比单纯按过去帧做滤波更少产生动作滞后。
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values.copy()

    forward = values.copy()
    for i in range(1, len(forward)):
        delta = np.clip(forward[i] - forward[i - 1], -max_delta, max_delta)
        forward[i] = forward[i - 1] + delta

    backward = values.copy()
    for i in range(len(backward) - 2, -1, -1):
        delta = np.clip(backward[i] - backward[i + 1], -max_delta, max_delta)
        backward[i] = backward[i + 1] + delta

    return 0.5 * (forward + backward)


def detect_spike_frames(
    dof_pos: np.ndarray,
    fps: int,
    joints: list[JointInfo],
) -> tuple[np.ndarray, dict[str, Any]]:
    # 把速度、加速度、jerk 超阈值的位置映射回原始帧，用于报告里定位
    # 哪些关节/帧最容易抽动。
    frame_mask = np.zeros_like(dof_pos, dtype=bool)
    velocity = derivative_abs(dof_pos, fps, order=1)
    acceleration = derivative_abs(dof_pos, fps, order=2)
    jerk = derivative_abs(dof_pos, fps, order=3)
    stats: dict[str, Any] = {
        "velocity_events": 0,
        "acceleration_events": 0,
        "jerk_events": 0,
        "per_joint_spike_frames": {},
    }

    derivative_specs = (
        ("velocity_events", velocity, 1, [joint.velocity_limit for joint in joints]),
        ("acceleration_events", acceleration, 2, [joint.accel_limit for joint in joints]),
        ("jerk_events", jerk, 3, [joint.jerk_limit for joint in joints]),
    )
    for event_key, derivative, order, limits in derivative_specs:
        if derivative.size == 0:
            continue
        for joint, limit in zip(joints, limits):
            spike_indices = np.flatnonzero(derivative[:, joint.dof_index] > float(limit))
            if len(spike_indices) == 0:
                continue
            radius = spike_radius_for_joint(joint.name, fps)
            stats[event_key] += int(len(spike_indices))
            for index in spike_indices:
                start = max(0, int(index) - radius)
                end = min(len(dof_pos), int(index) + order + radius + 1)
                frame_mask[start:end, joint.dof_index] = True

    for joint in joints:
        stats["per_joint_spike_frames"][joint.name] = int(
            np.count_nonzero(frame_mask[:, joint.dof_index])
        )
    stats["spike_frame_count"] = int(np.count_nonzero(np.any(frame_mask, axis=1)))
    stats["spike_value_count"] = int(np.count_nonzero(frame_mask))
    return frame_mask, stats


def spike_radius_for_joint(name: str, fps: int) -> int:
    if is_arm_joint(name):
        return max(1, int(round(0.04 * fps)))
    return max(1, int(round(0.02 * fps)))


def smooth_arm_joints_extra(
    dof_pos: np.ndarray,
    joints: list[JointInfo],
) -> tuple[np.ndarray, dict[str, Any]]:
    # 实测更稳的 V2 策略：先用 legacy 得到一个合法稳定的基底，
    # 再只对肩/肘/腕额外平滑。这样不会把下肢和 root 一起磨软。
    result = dof_pos.copy()
    stats: dict[str, Any] = {
        "arm_extra_smooth_joints": {},
        "arm_extra_smooth_joint_count": 0,
        "arm_extra_smooth_frames": 0,
    }
    for joint in joints:
        if not is_arm_joint(joint.name):
            continue
        window = min(joint.smooth_window + 8, 31)
        before = result[:, joint.dof_index].copy()
        result[:, joint.dof_index] = centered_smooth(before, window)
        changed = int(np.count_nonzero(np.abs(result[:, joint.dof_index] - before) > 1e-9))
        stats["arm_extra_smooth_joints"][joint.name] = {
            "window": int(odd_window(window, len(result))),
            "changed_frames": changed,
        }
        stats["arm_extra_smooth_joint_count"] += 1
        stats["arm_extra_smooth_frames"] += changed
    return result, stats


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


def fix_quat_hemisphere(quat_xyzw: np.ndarray) -> np.ndarray:
    # 四元数 q 和 -q 表示同一个旋转。先统一符号方向，可以避免根节点旋转
    # 平滑时出现“明明没转一圈但数值跳了 360 度”的假突变。
    quat = normalize_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).copy())
    for i in range(1, len(quat)):
        if float(np.dot(quat[i - 1], quat[i])) < 0.0:
            quat[i] *= -1.0
    return quat


def smooth_quat_xyzw(quat_xyzw: np.ndarray, window: int) -> np.ndarray:
    quat = fix_quat_hemisphere(quat_xyzw)
    window = odd_window(window, len(quat))
    if window <= 1:
        return quat

    weights = triangular_kernel(window)
    pad = window // 2
    result = np.empty_like(quat)
    for idx in range(len(quat)):
        lo = max(0, idx - pad)
        hi = min(len(quat), idx + pad + 1)
        local = quat[lo:hi].copy()
        center = quat[idx]
        for j in range(len(local)):
            if float(np.dot(center, local[j])) < 0.0:
                local[j] *= -1.0
        weight_lo = pad - (idx - lo)
        local_weights = weights[weight_lo : weight_lo + len(local)]
        result[idx] = (local * local_weights[:, None]).sum(axis=0)
    return fix_quat_hemisphere(result)


def quat_angle_diff_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = fix_quat_hemisphere(quat_xyzw)
    if len(quat) < 2:
        return np.zeros((0,), dtype=np.float64)
    dot = np.sum(quat[:-1] * quat[1:], axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_pair_angle_diff_xyzw(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> np.ndarray:
    first = fix_quat_hemisphere(first_xyzw)
    second = normalize_quat_xyzw(np.asarray(second_xyzw, dtype=np.float64))
    if len(first) != len(second):
        raise ValueError("Quaternion sequences must have the same length.")
    dot = np.sum(first * second, axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def clip_dof(dof_pos: np.ndarray, joints: list[JointInfo]) -> tuple[np.ndarray, int]:
    clipped = np.asarray(dof_pos, dtype=np.float64).copy()
    changed = 0
    for joint in joints:
        before = clipped[:, joint.dof_index].copy()
        clipped[:, joint.dof_index] = np.clip(before, joint.qrange[0], joint.qrange[1])
        changed += int(np.count_nonzero(np.abs(before - clipped[:, joint.dof_index]) > 1e-9))
    return clipped, changed


def optimize_motion(
    motion: dict[str, Any],
    *,
    robot: str,
    profile_name: str,
    pipeline_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pipeline_name == "legacy":
        return optimize_motion_legacy(motion, robot=robot, profile_name=profile_name)
    if pipeline_name == "v2":
        return optimize_motion_v2(motion, robot=robot, profile_name=profile_name)
    if pipeline_name == "v2_foot":
        return optimize_motion_v2_foot(motion, robot=robot, profile_name=profile_name)
    raise ValueError(f"Unsupported pipeline: {pipeline_name}")


def optimize_motion_legacy(
    motion: dict[str, Any],
    *,
    robot: str,
    profile_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = PROFILES[profile_name]
    model = load_robot_model(robot)
    fps, root_pos, root_rot, dof_pos = parse_motion_arrays(motion)
    joints = get_joint_info(model, dof_pos.shape[1], profile)
    before = quality_metrics(root_pos, root_rot, dof_pos, fps, joints, model)

    # 后处理主流程：
    # 1. 限制关节相邻帧突变；
    # 2. 按肩肘/腕/下肢/腰等关节组使用不同窗口做居中平滑；
    # 3. 根据 MuJoCo 模型关节范围做 clip；
    # 4. 最后平滑根节点位置和朝向。
    max_delta = np.asarray([joint.velocity_limit / fps for joint in joints], dtype=np.float64)
    limited_dof = velocity_limited_path(dof_pos, max_delta)

    smoothed_dof = limited_dof.copy()
    for joint in joints:
        smoothed_dof[:, joint.dof_index] = centered_smooth(
            limited_dof[:, joint.dof_index], joint.smooth_window
        )
    smoothed_dof, clipped_values = clip_dof(smoothed_dof, joints)

    smoothed_root_pos = centered_smooth(root_pos, profile.root_window)
    smoothed_root_rot = smooth_quat_xyzw(root_rot, profile.root_window)
    # 平滑根节点可能把整个人形机器人略微压低。这里保持原动作的最低脚底高度，
    # 避免后处理凭空引入脚穿地。
    ground_z_shift = preserve_min_foot_height(
        model,
        root_pos,
        root_rot,
        dof_pos,
        smoothed_root_pos,
        smoothed_root_rot,
        smoothed_dof,
    )

    optimized = dict(motion)
    optimized["root_pos"] = smoothed_root_pos.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
    optimized["root_rot"] = smoothed_root_rot.astype(np.asarray(motion["root_rot"]).dtype, copy=False)
    optimized["dof_pos"] = smoothed_dof.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)

    after = quality_metrics(smoothed_root_pos, smoothed_root_rot, smoothed_dof, fps, joints, model)
    report = build_report(
        robot=robot,
        profile_name=profile_name,
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        optimized_root_pos=smoothed_root_pos,
        optimized_root_rot=smoothed_root_rot,
        optimized_dof_pos=smoothed_dof,
        joints=joints,
        before=before,
        after=after,
        clipped_values=clipped_values,
        ground_z_shift=ground_z_shift,
        pipeline_name="legacy",
        optimizer_version="legacy_full_smooth",
        repair={},
    )
    return optimized, report


def optimize_motion_v2(
    motion: dict[str, Any],
    *,
    robot: str,
    profile_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = PROFILES[profile_name]
    model = load_robot_model(robot)
    fps, root_pos, root_rot, dof_pos = parse_motion_arrays(motion)
    joints = get_joint_info(model, dof_pos.shape[1], profile)
    before = quality_metrics(root_pos, root_rot, dof_pos, fps, joints, model)

    # V2 的重点是手臂抽动优先：先沿用 legacy 得到一个合法稳定的全身基底，
    # 再只对肩/肘/腕做额外平滑。这样比全身强平滑更能保留下肢和 root 节奏。
    spike_mask, spike_stats = detect_spike_frames(dof_pos, fps, joints)
    max_delta = np.asarray([joint.velocity_limit / fps for joint in joints], dtype=np.float64)
    limited_dof = velocity_limited_path(dof_pos, max_delta)

    stable_dof = limited_dof.copy()
    for joint in joints:
        stable_dof[:, joint.dof_index] = centered_smooth(
            limited_dof[:, joint.dof_index], joint.smooth_window
        )
    stable_dof, base_clipped_values = clip_dof(stable_dof, joints)

    arm_smoothed_dof, arm_smooth_stats = smooth_arm_joints_extra(stable_dof, joints)
    smoothed_dof, clipped_values = clip_dof(arm_smoothed_dof, joints)

    smoothed_root_pos = root_pos.copy()
    root_window = min(profile.root_window, 7)
    smoothed_root_pos[:, 2] = centered_smooth(root_pos[:, 2], root_window)
    smoothed_root_rot = smooth_quat_xyzw(root_rot, root_window)
    ground_z_shift = preserve_min_foot_height(
        model,
        root_pos,
        root_rot,
        dof_pos,
        smoothed_root_pos,
        smoothed_root_rot,
        smoothed_dof,
    )

    optimized = dict(motion)
    optimized["root_pos"] = smoothed_root_pos.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
    optimized["root_rot"] = smoothed_root_rot.astype(np.asarray(motion["root_rot"]).dtype, copy=False)
    optimized["dof_pos"] = smoothed_dof.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)

    after = quality_metrics(smoothed_root_pos, smoothed_root_rot, smoothed_dof, fps, joints, model)
    repair = {
        **spike_stats,
        **arm_smooth_stats,
        "base_clipped_values": int(base_clipped_values),
        "final_clipped_values": int(clipped_values),
        "arm_spike_value_count": int(
            sum(
                np.count_nonzero(spike_mask[:, joint.dof_index])
                for joint in joints
                if is_arm_joint(joint.name)
            )
        ),
    }
    report = build_report(
        robot=robot,
        profile_name=profile_name,
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        optimized_root_pos=smoothed_root_pos,
        optimized_root_rot=smoothed_root_rot,
        optimized_dof_pos=smoothed_dof,
        joints=joints,
        before=before,
        after=after,
        clipped_values=clipped_values,
        ground_z_shift=ground_z_shift,
        pipeline_name="v2",
        optimizer_version="v2_arm_spike",
        repair=repair,
    )
    return optimized, report


def optimize_motion_v2_foot(
    motion: dict[str, Any],
    *,
    robot: str,
    profile_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v2_motion, v2_report = optimize_motion_v2(motion, robot=robot, profile_name=profile_name)

    profile = PROFILES[profile_name]
    model = load_robot_model(robot)
    fps, root_pos, root_rot, dof_pos = parse_motion_arrays(motion)
    joints = get_joint_info(model, dof_pos.shape[1], profile)
    _, v2_root_pos, v2_root_rot, v2_dof_pos = parse_motion_arrays(v2_motion)

    corrected_root_pos, foot_contact, foot_lock = apply_light_foot_lock(
        model,
        v2_root_pos,
        v2_root_rot,
        v2_dof_pos,
        fps,
    )

    optimized = dict(v2_motion)
    optimized["root_pos"] = corrected_root_pos.astype(
        np.asarray(v2_motion["root_pos"]).dtype, copy=False
    )

    after = quality_metrics(corrected_root_pos, v2_root_rot, v2_dof_pos, fps, joints, model)
    repair = {
        **v2_report.get("repair", {}),
        "foot_lock_available": bool(foot_lock.get("available", False)),
        "foot_lock_corrected_frames": int(foot_lock.get("corrected_frames", 0)),
    }
    report = build_report(
        robot=robot,
        profile_name=profile_name,
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        optimized_root_pos=corrected_root_pos,
        optimized_root_rot=v2_root_rot,
        optimized_dof_pos=v2_dof_pos,
        joints=joints,
        before=v2_report["before"],
        after=after,
        clipped_values=int(v2_report.get("clipped_values", 0)),
        ground_z_shift=float(v2_report.get("ground_z_shift", 0.0)),
        pipeline_name="v2_foot",
        optimizer_version="v3_light_foot_lock",
        repair=repair,
    )
    report["foot_contact"] = foot_contact
    report["foot_lock"] = foot_lock
    return optimized, report


def quality_only(motion: dict[str, Any], *, robot: str) -> dict[str, Any]:
    profile = PROFILES["preview"]
    model = load_robot_model(robot)
    fps, root_pos, root_rot, dof_pos = parse_motion_arrays(motion)
    joints = get_joint_info(model, dof_pos.shape[1], profile)
    metrics = quality_metrics(root_pos, root_rot, dof_pos, fps, joints, model)
    return {
        "robot": robot,
        "mode": "quality",
        "frames": int(len(dof_pos)),
        "fps": fps,
        "joint_count": int(dof_pos.shape[1]),
        "metrics": metrics,
    }


def build_report(
    *,
    robot: str,
    profile_name: str,
    pipeline_name: str,
    optimizer_version: str,
    fps: int,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    optimized_root_pos: np.ndarray,
    optimized_root_rot: np.ndarray,
    optimized_dof_pos: np.ndarray,
    joints: list[JointInfo],
    before: dict[str, Any],
    after: dict[str, Any],
    clipped_values: int,
    ground_z_shift: float,
    repair: dict[str, Any],
) -> dict[str, Any]:
    root_rot_delta = quat_pair_angle_diff_xyzw(root_rot, optimized_root_rot)
    return {
        "robot": robot,
        "mode": "optimize",
        "profile": profile_name,
        "pipeline": pipeline_name,
        "optimizer_version": optimizer_version,
        "frames": int(len(dof_pos)),
        "fps": fps,
        "joint_count": int(dof_pos.shape[1]),
        "joint_names": [joint.name for joint in joints],
        "clipped_values": int(clipped_values),
        "ground_z_shift": float(ground_z_shift),
        "repair": repair,
        "change": {
            "mean_abs_dof_change": float(np.mean(np.abs(optimized_dof_pos - dof_pos))),
            "max_abs_dof_change": max_value(np.abs(optimized_dof_pos - dof_pos)),
            "mean_root_pos_change_m": float(np.mean(np.linalg.norm(optimized_root_pos - root_pos, axis=1))),
            "max_root_pos_change_m": max_value(np.linalg.norm(optimized_root_pos - root_pos, axis=1)),
            "root_rot_change_rad_approx": max_value(root_rot_delta),
        },
        "before": before,
        "after": after,
    }


def quality_metrics(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    fps: int,
    joints: list[JointInfo],
    model: mj.MjModel,
) -> dict[str, Any]:
    # 速度用于发现瞬间大跳， 加速度用于发现方向急变，jerk 对视觉上的“抽一下”
    # 特别敏感，所以报告里三类指标都会保留。
    dof_velocity = derivative_abs(dof_pos, fps, order=1)
    dof_accel = derivative_abs(dof_pos, fps, order=2)
    dof_jerk = derivative_abs(dof_pos, fps, order=3)
    root_velocity = derivative_vector_norm(root_pos, fps, order=1)
    root_accel = derivative_vector_norm(root_pos, fps, order=2)
    root_jerk = derivative_vector_norm(root_pos, fps, order=3)
    root_angular_velocity = quat_angle_diff_xyzw(root_rot) * fps
    root_angular_accel = derivative_abs(root_angular_velocity[:, None], fps, order=1).reshape(-1)

    per_joint = {}
    joint_limit_violations = 0
    min_joint_limit_margin = float("inf")
    for joint in joints:
        idx = joint.dof_index
        values = dof_pos[:, idx]
        below = values < joint.qrange[0]
        above = values > joint.qrange[1]
        joint_limit_violations += int(np.count_nonzero(below | above))
        margin = np.minimum(values - joint.qrange[0], joint.qrange[1] - values)
        min_joint_limit_margin = min(min_joint_limit_margin, min_value(margin))
        per_joint[joint.name] = {
            "velocity_max": column_max(dof_velocity, idx),
            "acceleration_max": column_max(dof_accel, idx),
            "jerk_max": column_max(dof_jerk, idx),
            "velocity_spikes": column_spikes(dof_velocity, idx, joint.velocity_limit),
            "acceleration_spikes": column_spikes(dof_accel, idx, joint.accel_limit),
            "jerk_spikes": column_spikes(dof_jerk, idx, joint.jerk_limit),
            "limit_violations": int(np.count_nonzero(below | above)),
            "min_limit_margin": min_value(margin),
        }

    top_velocity_joints = top_joint_metric(per_joint, "velocity_max")
    top_acceleration_joints = top_joint_metric(per_joint, "acceleration_max")
    top_jerk_joints = top_joint_metric(per_joint, "jerk_max")
    contacts = contact_metrics(root_pos, root_rot, dof_pos, fps, model)

    return {
        "dof_velocity": summarize_array(dof_velocity),
        "dof_acceleration": summarize_array(dof_accel),
        "dof_jerk": summarize_array(dof_jerk),
        "root_velocity": summarize_array(root_velocity),
        "root_acceleration": summarize_array(root_accel),
        "root_jerk": summarize_array(root_jerk),
        "root_angular_velocity": summarize_array(root_angular_velocity),
        "root_angular_acceleration": summarize_array(root_angular_accel),
        "joint_limit_violations": int(joint_limit_violations),
        "min_joint_limit_margin": (
            0.0 if not np.isfinite(min_joint_limit_margin) else float(min_joint_limit_margin)
        ),
        "spike_count_total": int(
            sum(item["velocity_spikes"] + item["acceleration_spikes"] + item["jerk_spikes"] for item in per_joint.values())
        ),
        "velocity_spike_count_total": int(sum(item["velocity_spikes"] for item in per_joint.values())),
        "acceleration_spike_count_total": int(sum(item["acceleration_spikes"] for item in per_joint.values())),
        "jerk_spike_count_total": int(sum(item["jerk_spikes"] for item in per_joint.values())),
        "top_velocity_joints": top_velocity_joints,
        "top_acceleration_joints": top_acceleration_joints,
        "top_jerk_joints": top_jerk_joints,
        "contact": contacts,
        "per_joint": per_joint,
    }


def derivative_abs(values: np.ndarray, fps: int, *, order: int) -> np.ndarray:
    if len(values) <= order:
        return np.zeros((0,) + values.shape[1:], dtype=np.float64)
    return np.abs(np.diff(values, n=order, axis=0)) * (fps**order)


def derivative_vector_norm(values: np.ndarray, fps: int, *, order: int) -> np.ndarray:
    if len(values) <= order:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(values, n=order, axis=0), axis=1) * (fps**order)


def summarize_array(values: np.ndarray) -> dict[str, float]:
    return {
        "max": max_value(values),
        "p95": percentile(values, 95),
        "rms": rms(values),
    }


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def max_value(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def min_value(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def column_max(values: np.ndarray, index: int) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values[:, index]))


def column_spikes(values: np.ndarray, index: int, limit: float) -> int:
    if values.size == 0:
        return 0
    return int(np.count_nonzero(values[:, index] > limit))


def top_joint_metric(per_joint: dict[str, dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    items = sorted(
        ((name, float(values[metric])) for name, values in per_joint.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    return [{"joint": name, metric: value} for name, value in items]


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for index in indices[1:]:
        index = int(index)
        if index == previous + 1:
            previous = index
            continue
        segments.append((start, previous))
        start = previous = index
    segments.append((start, previous))
    return segments


def close_boolean_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    segments = mask_to_segments(result)
    for (_, previous_end), (next_start, _) in zip(segments, segments[1:]):
        gap = next_start - previous_end - 1
        if 0 < gap <= max_gap:
            result[previous_end + 1 : next_start] = True
    return result


def remove_short_segments(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in mask_to_segments(result):
        if end - start + 1 < min_length:
            result[start : end + 1] = False
    return result


def find_foot_body_ids(model: mj.MjModel) -> list[int]:
    ids = []
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if name and any(hint in name.lower() for hint in FOOT_NAME_HINTS):
            ids.append(body_id)
    return ids


def find_foot_geom_ids(model: mj.MjModel) -> dict[str, list[int]]:
    foot_geoms = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        lower = name.lower()
        if not ("foot" in lower and "collision" in lower):
            continue
        if lower.startswith("l_"):
            foot_geoms["left"].append(geom_id)
        elif lower.startswith("r_"):
            foot_geoms["right"].append(geom_id)
    return {side: ids for side, ids in foot_geoms.items() if ids}


def replay_foot_geom_tracks(
    model: mj.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    # 用 MuJoCo 正运动学重放动作，直接读取 XML 里 foot collision geom 的世界坐标。
    # 这比用 ankle body 近似脚底更接近真实接触点，也更适合判断脚滑。
    foot_geom_ids = find_foot_geom_ids(model)
    if not foot_geom_ids:
        return {}, {}

    data = mj.MjData(model)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    tracks = {
        side: np.zeros((len(root_pos), len(ids), 3), dtype=np.float64)
        for side, ids in foot_geom_ids.items()
    }
    names = {
        side: [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id) for geom_id in ids]
        for side, ids in foot_geom_ids.items()
    }

    for frame_idx in range(len(root_pos)):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot_wxyz[frame_idx]
        data.qpos[7:] = dof_pos[frame_idx]
        mj.mj_forward(model, data)
        for side, ids in foot_geom_ids.items():
            for local_idx, geom_id in enumerate(ids):
                tracks[side][frame_idx, local_idx] = data.geom_xpos[geom_id]
    return tracks, names


def foot_representative_track(positions: np.ndarray) -> np.ndarray:
    # 多个 foot collision capsule 属于同一只脚。XY 用平均位置更抗噪，
    # Z 用最低点更接近“有没有贴近地面”。
    xy = np.mean(positions[:, :, :2], axis=1)
    z = np.min(positions[:, :, 2], axis=1)
    return np.column_stack([xy, z])


def build_foot_contact_masks(
    tracks: dict[str, np.ndarray],
    fps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # 接触检测不是物理仿真，只是一个轻量启发式：
    # 脚够低、竖直速度不大、水平速度不过分离谱时，认为它可能是支撑脚。
    # 后面会合并很短的断点，避免一个支撑段被噪声切成很多碎片。
    masks: dict[str, np.ndarray] = {}
    side_stats: dict[str, Any] = {}
    max_gap = max(1, int(round(0.06 * fps)))
    min_segment = max(3, int(round(0.10 * fps)))

    for side, positions in tracks.items():
        representative = foot_representative_track(positions)
        z = representative[:, 2]
        xy_speed = (
            np.linalg.norm(np.diff(representative[:, :2], axis=0), axis=1) * fps
            if len(representative) > 1
            else np.zeros((0,), dtype=np.float64)
        )
        z_speed = (
            np.abs(np.diff(z, prepend=z[0])) * fps
            if len(z)
            else np.zeros((0,), dtype=np.float64)
        )
        xy_speed_full = np.concatenate([[0.0], xy_speed]) if xy_speed.size else np.zeros_like(z)
        height_threshold = percentile(z, 8) + 0.035
        raw_mask = (z <= height_threshold) & (z_speed <= 1.0) & (xy_speed_full <= 3.0)
        contact_mask = remove_short_segments(close_boolean_gaps(raw_mask, max_gap), min_segment)
        masks[side] = contact_mask
        side_stats[side] = {
            "contact_frames": int(np.count_nonzero(contact_mask)),
            "contact_segments": int(len(mask_to_segments(contact_mask))),
            "height_threshold": float(height_threshold),
            "min_height": min_value(z),
            "xy_speed_p95_when_contact": (
                percentile(xy_speed[contact_mask[:-1]], 95)
                if xy_speed.size and np.any(contact_mask[:-1])
                else 0.0
            ),
        }
    return masks, side_stats


def summarize_foot_geom_contact(
    tracks: dict[str, np.ndarray],
    names: dict[str, list[str]],
    fps: int,
) -> dict[str, Any]:
    masks, side_stats = build_foot_contact_masks(tracks, fps)
    sliding_values = []
    per_frame_lowest = []
    min_height = float("inf")
    for side, positions in tracks.items():
        representative = foot_representative_track(positions)
        min_height = min(min_height, min_value(representative[:, 2]))
        per_frame_lowest.append(representative[:, 2])
        if len(representative) > 1 and np.any(masks[side][:-1]):
            xy_speed = np.linalg.norm(np.diff(representative[:, :2], axis=0), axis=1) * fps
            sliding_values.append(xy_speed[masks[side][:-1]])

    all_sliding = np.concatenate(sliding_values) if sliding_values else np.zeros((0,), dtype=np.float64)
    lowest = np.min(np.column_stack(per_frame_lowest), axis=1) if per_frame_lowest else np.zeros((0,))
    if not np.isfinite(min_height):
        min_height = 0.0
    return {
        "available": True,
        "source": "foot_collision_geoms",
        "foot_geoms": names,
        "min_foot_height": float(min_height),
        "estimated_ground_penetration_depth": max(0.0, -float(min_height)),
        "lowest_foot_height_p05": percentile(lowest, 5),
        "estimated_foot_sliding_speed": summarize_array(all_sliding),
        "side_metrics": side_stats,
    }


def clip_vector_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64).copy()
    norm = np.linalg.norm(result, axis=1)
    too_large = norm > max_norm
    if np.any(too_large):
        result[too_large] *= (max_norm / np.maximum(norm[too_large], 1e-8))[:, None]
    return result


def edge_taper(length: int, ramp: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.float64)
    ramp = max(1, min(int(ramp), max(1, length // 2)))
    weights = np.ones((length,), dtype=np.float64)
    weights[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=True)
    weights[-ramp:] = np.minimum(weights[-ramp:], np.linspace(1.0, 0.0, ramp, endpoint=True))
    return weights


def limit_vector_step(values: np.ndarray, max_step: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for idx in range(1, len(result)):
        delta = result[idx] - result[idx - 1]
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            result[idx] = result[idx - 1] + delta / max(norm, 1e-8) * max_step
    return result


def apply_light_foot_lock(
    model: mj.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    fps: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    # Foot-lock 的核心思路：
    # 1. 找到支撑脚接触段；
    # 2. 用接触段内脚底 XY 的中位数作为“应该钉住”的支撑位置；
    # 3. 如果后续脚底在地上滑，就反向微调 root XY 抵消漂移。
    # 注意这里不改 dof_pos，不改 root Z，因此不会破坏 V2 的手臂平滑结果。
    foot_tracks, foot_names = replay_foot_geom_tracks(model, root_pos, root_rot_xyzw, dof_pos)
    if not foot_tracks:
        return (
            root_pos.copy(),
            {"available": False, "reason": "No foot collision geoms found."},
            {"available": False, "reason": "No foot collision geoms found."},
        )

    contact_masks, contact_side_stats = build_foot_contact_masks(foot_tracks, fps)
    correction_sum = np.zeros((len(root_pos), 2), dtype=np.float64)
    correction_weight = np.zeros((len(root_pos),), dtype=np.float64)

    strength = 0.65
    max_segment_correction = 0.08
    max_step = 0.006
    smooth_window = max(3, int(round(0.12 * fps)))
    ramp_frames = max(2, int(round(0.08 * fps)))
    total_segments = 0
    skipped_segments = 0

    for side, positions in foot_tracks.items():
        representative = foot_representative_track(positions)
        for start, end in mask_to_segments(contact_masks[side]):
            length = end - start + 1
            if length < max(3, int(round(0.10 * fps))):
                skipped_segments += 1
                continue
            total_segments += 1
            # 如果只取接触段开头做锚点，开头刚好有噪声时会把整段都带偏。
            # 这里优先用整段脚底 XY 的中位数做“支撑点”，比均值更不怕偶发跳点。
            anchor_xy = np.median(representative[start : end + 1, :2], axis=0)
            raw_correction = anchor_xy[None, :] - representative[start : end + 1, :2]
            raw_correction = clip_vector_norm(raw_correction, max_segment_correction)
            raw_correction = centered_smooth(raw_correction, smooth_window)
            taper = edge_taper(length, ramp_frames)
            correction = raw_correction * taper[:, None] * strength
            correction_sum[start : end + 1] += correction
            correction_weight[start : end + 1] += 1.0

    combined = np.zeros_like(correction_sum)
    active = correction_weight > 0
    combined[active] = correction_sum[active] / correction_weight[active, None]
    combined = centered_smooth(combined, smooth_window)
    combined = limit_vector_step(combined, max_step)

    corrected_root_pos = root_pos.copy()
    corrected_root_pos[:, :2] += combined
    correction_norm = np.linalg.norm(combined, axis=1)
    contact_info = {
        "available": True,
        "source": "foot_collision_geoms",
        "foot_geoms": foot_names,
        "sides": contact_side_stats,
    }
    foot_lock_info = {
        "available": True,
        "strength": strength,
        "max_segment_correction_m": max_segment_correction,
        "max_step_m_per_frame": max_step,
        "corrected_frames": int(np.count_nonzero(correction_norm > 1e-6)),
        "contact_segments_used": int(total_segments),
        "contact_segments_skipped": int(skipped_segments),
        "mean_root_xy_correction_m": float(np.mean(correction_norm)),
        "max_root_xy_correction_m": max_value(correction_norm),
        "p95_root_xy_correction_m": percentile(correction_norm, 95),
    }
    return corrected_root_pos, contact_info, foot_lock_info


def contact_metrics(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    fps: int,
    model: mj.MjModel,
) -> dict[str, Any]:
    foot_tracks, foot_names = replay_foot_geom_tracks(model, root_pos, root_rot_xyzw, dof_pos)
    if foot_tracks:
        return summarize_foot_geom_contact(foot_tracks, foot_names, fps)

    foot_body_ids = find_foot_body_ids(model)
    if not foot_body_ids:
        return {"available": False, "reason": "No foot/toe/sole/ankle body found in MuJoCo model."}

    data = mj.MjData(model)
    foot_positions = np.zeros((len(root_pos), len(foot_body_ids), 3), dtype=np.float64)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    # 通过 MuJoCo 正运动学重放整段动作，估计脚底高度和脚滑速度。
    # 这样不需要原始 BVH/SMPL 数据里提供接触标签。
    for idx in range(len(root_pos)):
        data.qpos[:3] = root_pos[idx]
        data.qpos[3:7] = root_rot_wxyz[idx]
        data.qpos[7:] = dof_pos[idx]
        mj.mj_forward(model, data)
        for foot_idx, body_id in enumerate(foot_body_ids):
            foot_positions[idx, foot_idx] = data.xpos[body_id]

    foot_z = foot_positions[:, :, 2]
    min_height = float(np.min(foot_z)) if foot_z.size else 0.0
    penetration_depth = max(0.0, -min_height)
    per_frame_lowest = np.min(foot_z, axis=1)
    near_ground = foot_z <= (min_height + 0.03)
    xy_speed = (
        np.linalg.norm(np.diff(foot_positions[:, :, :2], axis=0), axis=2) * fps
        if len(root_pos) > 1
        else np.zeros((0, len(foot_body_ids)), dtype=np.float64)
    )
    if xy_speed.size:
        contact_mask = near_ground[:-1]
        sliding_values = xy_speed[contact_mask]
    else:
        sliding_values = np.zeros((0,), dtype=np.float64)

    return {
        "available": True,
        "foot_bodies": [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) for body_id in foot_body_ids
        ],
        "min_foot_height": min_height,
        "estimated_ground_penetration_depth": penetration_depth,
        "lowest_foot_height_p05": percentile(per_frame_lowest, 5),
        "estimated_foot_sliding_speed": summarize_array(sliding_values),
    }


def preserve_min_foot_height(
    model: mj.MjModel,
    original_root_pos: np.ndarray,
    original_root_rot_xyzw: np.ndarray,
    original_dof_pos: np.ndarray,
    optimized_root_pos: np.ndarray,
    optimized_root_rot_xyzw: np.ndarray,
    optimized_dof_pos: np.ndarray,
) -> float:
    # 轻量地面修正：只有当优化后的最低脚底高度低于原动作时，
    # 才整体抬高 root z；这不是完整接触优化，只是避免预览穿地。
    original_min = min_foot_height(model, original_root_pos, original_root_rot_xyzw, original_dof_pos)
    optimized_min = min_foot_height(model, optimized_root_pos, optimized_root_rot_xyzw, optimized_dof_pos)
    if original_min is None or optimized_min is None or optimized_min >= original_min:
        return 0.0
    shift = float(original_min - optimized_min)
    optimized_root_pos[:, 2] += shift
    return shift


def min_foot_height(
    model: mj.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> float | None:
    foot_body_ids = find_foot_body_ids(model)
    if not foot_body_ids:
        return None
    data = mj.MjData(model)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    min_height = float("inf")
    for idx in range(len(root_pos)):
        data.qpos[:3] = root_pos[idx]
        data.qpos[3:7] = root_rot_wxyz[idx]
        data.qpos[7:] = dof_pos[idx]
        mj.mj_forward(model, data)
        for body_id in foot_body_ids:
            min_height = min(min_height, float(data.xpos[body_id, 2]))
    return min_height if np.isfinite(min_height) else None


def default_quality_output(input_path: Path) -> Path:
    return input_path.with_name("motion_quality.json")


def output_name_label(profile: str, pipeline: str) -> str:
    # 目录本身已经区分任务，文件名就尽量短一点。
    # 当前推荐组合 soft + v2_foot 直接叫 foot，避免 robot_motion_soft_foot_smooth 这种长串。
    if pipeline == "v2_foot":
        return "foot" if profile == "soft" else f"{profile}_foot"
    if pipeline == "legacy":
        return f"{profile}_legacy"
    return profile


def default_optimize_output(input_path: Path, profile: str, pipeline: str) -> Path:
    return input_path.with_name(f"motion_{output_name_label(profile, pipeline)}.pkl")


def default_optimize_quality_output(input_path: Path, profile: str, pipeline: str) -> Path:
    return input_path.with_name(f"quality_{output_name_label(profile, pipeline)}.json")


def default_video_output(input_path: Path, profile: str, pipeline: str) -> Path:
    return input_path.with_name(f"preview_{output_name_label(profile, pipeline)}.mp4")


def run_quality(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve() if args.output else default_quality_output(input_path)
    )
    motion = load_motion(input_path)
    report = quality_only(motion, robot=args.robot)
    report["input"] = str(input_path)
    write_report(output_path, report)
    metrics = report["metrics"]
    print(f"[OK] Saved quality report: {output_path}")
    print(f"[Quality] dof_velocity_max: {metrics['dof_velocity']['max']:.3f} rad/s")
    print(f"[Quality] dof_acceleration_max: {metrics['dof_acceleration']['max']:.3f} rad/s^2")
    print(f"[Quality] dof_jerk_max: {metrics['dof_jerk']['max']:.3f} rad/s^3")
    print(f"[Quality] spike_count_total: {metrics['spike_count_total']}")


def run_optimize(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_optimize_output(input_path, args.profile, args.pipeline)
    )
    quality_path = (
        Path(args.quality_json).expanduser().resolve()
        if args.quality_json
        else default_optimize_quality_output(input_path, args.profile, args.pipeline)
    )
    video_path = (
        Path(args.video_output).expanduser().resolve()
        if args.video_output
        else default_video_output(input_path, args.profile, args.pipeline)
    )

    if input_path == output_path:
        raise ValueError("Refusing to overwrite the input pkl. Please choose a different output path.")

    motion = load_motion(input_path)
    optimized, report = optimize_motion(
        motion,
        robot=args.robot,
        profile_name=args.profile,
        pipeline_name=args.pipeline,
    )
    report["input"] = str(input_path)
    report["output"] = str(output_path)
    save_motion(output_path, optimized)
    write_report(quality_path, report)

    print(f"[OK] Saved optimized motion: {output_path}")
    print(f"[OK] Saved quality report: {quality_path}")
    print(
        "[Quality] dof_velocity_max: "
        f"{report['before']['dof_velocity']['max']:.3f} -> {report['after']['dof_velocity']['max']:.3f} rad/s"
    )
    print(
        "[Quality] dof_acceleration_max: "
        f"{report['before']['dof_acceleration']['max']:.3f} -> {report['after']['dof_acceleration']['max']:.3f} rad/s^2"
    )
    print(
        "[Quality] dof_jerk_max: "
        f"{report['before']['dof_jerk']['max']:.3f} -> {report['after']['dof_jerk']['max']:.3f} rad/s^3"
    )
    print(
        "[Quality] spike_count_total: "
        f"{report['before']['spike_count_total']} -> {report['after']['spike_count_total']}"
    )

    if args.render:
        render_robot_motion(
            output_path,
            video_path,
            robot=args.robot,
            width=args.width,
            height=args.height,
            logger=print,
        )
        print(f"[OK] Saved preview video: {video_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quality":
        run_quality(args)
    elif args.command == "optimize":
        run_optimize(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
