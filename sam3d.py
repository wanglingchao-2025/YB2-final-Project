import sys
import os
import json
import shutil
import numpy as np
import open3d as o3d

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_temp_indexed_masks(items, mask_dir, temp_mask_dir):
    os.makedirs(temp_mask_dir, exist_ok=True)

    for f in os.listdir(temp_mask_dir):
        fp = os.path.join(temp_mask_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    for idx, item in enumerate(items):
        src_name = item["mask_file"]
        src_path = os.path.join(mask_dir, src_name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Mask file not found: {src_path}")
        dst_path = os.path.join(temp_mask_dir, f"{idx}.png")
        shutil.copy2(src_path, dst_path)

    print(f"[INFO] Temporary indexed masks prepared at: {temp_mask_dir}")


def make_point_cloud(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pc


def voxel_downsample_np(points, voxel_size):
    if len(points) == 0:
        return points
    pc = make_point_cloud(points)
    pc_ds = pc.voxel_down_sample(voxel_size)
    return np.asarray(pc_ds.points)


def robust_chamfer_distance_np(src_pts, tgt_pts, trim_ratio=0.90):
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return 1e9

    pc1 = make_point_cloud(src_pts)
    pc2 = make_point_cloud(tgt_pts)

    d1 = np.asarray(pc1.compute_point_cloud_distance(pc2))
    d2 = np.asarray(pc2.compute_point_cloud_distance(pc1))

    if len(d1) == 0 or len(d2) == 0:
        return 1e9

    k1 = max(1, int(len(d1) * trim_ratio))
    k2 = max(1, int(len(d2) * trim_ratio))
    d1 = np.partition(d1, k1 - 1)[:k1]
    d2 = np.partition(d2, k2 - 1)[:k2]
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


# =========================================================
# 点云清洗：只保留主体，尽量避免过度删除
# =========================================================
def filter_finite_points(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def bbox_diag(points):
    if len(points) == 0:
        return 0.0
    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    return float(np.linalg.norm(mx - mn))


def statistical_filter_np(points, nb_neighbors=24, std_ratio=2.5):
    if len(points) < max(nb_neighbors + 1, 20):
        return points
    pc = make_point_cloud(points)
    pc_filtered, _ = pc.remove_statistical_outlier(
        nb_neighbors=min(nb_neighbors, max(5, len(points) - 1)),
        std_ratio=std_ratio,
    )
    return np.asarray(pc_filtered.points)


def radius_filter_np(points, nb_points=10, radius=0.05):
    if len(points) < max(nb_points + 1, 20):
        return points
    pc = make_point_cloud(points)
    pc_filtered, _ = pc.remove_radius_outlier(
        nb_points=max(4, nb_points),
        radius=max(radius, 1e-6),
    )
    return np.asarray(pc_filtered.points)


def largest_cluster_np(points, eps=0.05, min_points=20):
    if len(points) < max(min_points + 1, 30):
        return points

    pc = make_point_cloud(points)
    labels = np.array(pc.cluster_dbscan(
        eps=max(eps, 1e-6),
        min_points=max(4, min_points),
        print_progress=False
    ))

    if len(labels) == 0 or np.max(labels) < 0:
        return points

    valid = labels >= 0
    if not np.any(valid):
        return points

    major = np.argmax(np.bincount(labels[valid]))
    keep = labels == major
    pts = np.asarray(pc.points)[keep]
    return pts


def safe_keep(prev_pts, cand_pts, min_ratio=0.20, min_points=150):
    """
    防止清洗过猛：
    如果新点云太少，就退回上一步结果。
    """
    if len(prev_pts) == 0:
        return cand_pts

    threshold = min(len(prev_pts), max(min_points, int(len(prev_pts) * min_ratio)))
    if len(cand_pts) >= threshold:
        return cand_pts
    return prev_pts


def clean_point_cloud_np(points, voxel_size=0.008):
    """
    更稳的清洗策略：
    1) 去掉 NaN/Inf
    2) 轻度体素下采样
    3) 统计滤波
    4) 半径滤波
    5) 只保留最大连通簇
    6) 每一步都做 fallback，避免把主体删没
    """
    pts = filter_finite_points(points)
    if len(pts) == 0:
        return pts

    # Stage 0: 原始有效点
    current = pts.copy()

    # Stage 1: 轻度体素下采样
    ds = voxel_downsample_np(current, voxel_size=voxel_size)
    current = safe_keep(current, ds, min_ratio=0.15, min_points=300)

    # 估计尺度，给后面的 radius / cluster 自适应参数
    diag = bbox_diag(current)
    if diag <= 1e-9:
        return current

    # Stage 2: 统计滤波（更保守一点，尽量保留细节）
    cand = statistical_filter_np(
        current,
        nb_neighbors=24,
        std_ratio=2.5
    )
    current = safe_keep(current, cand, min_ratio=0.35, min_points=250)

    # Stage 3: 半径滤波（半径随物体尺寸变化）
    adaptive_radius = max(voxel_size * 5.0, diag * 0.015)
    cand = radius_filter_np(
        current,
        nb_points=10,
        radius=adaptive_radius
    )
    current = safe_keep(current, cand, min_ratio=0.35, min_points=200)

    # Stage 4: 最大连通簇（保留主体）
    adaptive_eps = max(voxel_size * 6.0, diag * 0.02)
    cand = largest_cluster_np(
        current,
        eps=adaptive_eps,
        min_points=20
    )
    current = safe_keep(current, cand, min_ratio=0.25, min_points=180)

    return current


def compute_bbox_init(src_pts, tgt_pts):
    src_pc = make_point_cloud(src_pts)
    tgt_pc = make_point_cloud(tgt_pts)

    sb = src_pc.get_axis_aligned_bounding_box()
    tb = tgt_pc.get_axis_aligned_bounding_box()

    s_extent = np.asarray(sb.get_extent()) + 1e-9
    t_extent = np.asarray(tb.get_extent()) + 1e-9
    scale = float(np.median(t_extent / s_extent))

    s_center = np.asarray(sb.get_center())
    t_center = np.asarray(tb.get_center())
    trans = t_center - scale * s_center
    return scale, trans


def apply_axis_transform(points, perm=(0, 1, 2), sign=(1, 1, 1)):
    pts = points[:, perm].copy()
    pts *= np.array(sign, dtype=np.float64)[None, :]
    return pts


def yaw_rotation(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def apply_sim3(points, scale, yaw_deg, trans, perm=(0, 1, 2), sign=(1, 1, 1)):
    pts = apply_axis_transform(points, perm=perm, sign=sign)
    R = yaw_rotation(yaw_deg)
    return scale * (pts @ R.T) + trans[None, :]


def coarse_cd_score_to_anchor(src_pts, anchor_pts):
    """
    用于 seed 选择的轻量评分，不是最终对齐。
    尝试少量 perm/sign/yaw，取最小 trimmed CD。
    """
    if len(src_pts) == 0 or len(anchor_pts) == 0:
        return 1e9, {"perm": (0, 1, 2), "sign": (1, 1, 1), "yaw_deg": 0.0}

    src_ds = voxel_downsample_np(src_pts, 0.02)
    tgt_ds = voxel_downsample_np(anchor_pts, 0.02)

    perms = list(__import__("itertools").permutations([0, 1, 2]))
    signs = list(__import__("itertools").product([1, -1], repeat=3))
    yaw_candidates = [-180, -90, 0, 90, 180]

    best_cd = 1e9
    best_cfg = None

    for perm in perms:
        for sign in signs:
            x = apply_axis_transform(src_ds, perm=perm, sign=sign)
            for yaw_deg in yaw_candidates:
                R = yaw_rotation(yaw_deg)
                x2 = x @ R.T
                scale, trans = compute_bbox_init(x2, tgt_ds)
                x3 = scale * x2 + trans[None, :]
                cd = robust_chamfer_distance_np(x3, tgt_ds, trim_ratio=0.90)
                if cd < best_cd:
                    best_cd = cd
                    best_cfg = {
                        "perm": perm,
                        "sign": sign,
                        "yaw_deg": float(yaw_deg),
                        "scale": float(scale),
                        "trans": trans.tolist(),
                    }

    return best_cd, best_cfg


def reconstruct_multi_objects_bestofn(
    sam3_manifest_path,
    anchor_meta_path=None,
    output_dir="sam_ply",
    temp_mask_dir="_temp_sam3d_masks",
    num_seeds=2,
    base_seed=42,
    keep_all_candidates=False,
):
    os.makedirs(output_dir, exist_ok=True)

    sam3_manifest = load_json(sam3_manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(sam3_manifest_path))

    image_path = sam3_manifest.get("saved_image") or sam3_manifest.get("image_path")
    if image_path is None:
        raise ValueError("SAM3 manifest must contain 'saved_image' or 'image_path'")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask_dir = manifest_dir
    items = sorted(sam3_manifest["items"], key=lambda x: x["object_id"])
    mask_files = [item["mask_file"] for item in items]

    # anchor 可选，用于 seed 打分
    anchor_map = {}
    if anchor_meta_path is not None:
        anchor_meta = load_json(anchor_meta_path)
        anchor_dir = os.path.dirname(os.path.abspath(anchor_meta_path))
        for item in anchor_meta["items"]:
            oid = int(item["object_id"])
            anchor_path = os.path.join(anchor_dir, item["anchor_file"])
            if os.path.exists(anchor_path):
                anchor_map[oid] = anchor_path
        print(f"[INFO] Loaded {len(anchor_map)} anchors for SAM3D seed scoring.")

    prepare_temp_indexed_masks(items, mask_dir, temp_mask_dir)

    print("加载 SAM3D 模型...")
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    print("SAM3D 模型加载完成")

    print("加载图像...")
    image = load_image(image_path)

    print(f"检测到 {len(items)} 个 mask")
    print(f"每个 mask 将尝试 {num_seeds} 个 seeds")

    results = []
    candidate_root = os.path.join(output_dir, "candidates")
    if keep_all_candidates:
        os.makedirs(candidate_root, exist_ok=True)

    for idx, item in enumerate(items):
        oid = int(item["object_id"])
        mf = item["mask_file"]

        print(f"\n========== Object {oid:03d} / mask={mf} ==========")

        mask = load_single_mask(temp_mask_dir, index=idx)

        best = None
        seed_logs = []

        anchor_pts = None
        if oid in anchor_map:
            anchor_pts = np.asarray(np.load(anchor_map[oid]), dtype=np.float64)

        for s in range(num_seeds):
            seed = base_seed + s

            print(f"[SEED {seed}] reconstructing...")
            output = inference(image, mask, seed=seed)

            tmp_ply = os.path.join(output_dir, f"_tmp_object_{oid:03d}_seed_{seed}.ply")
            output["gs"].save_ply(tmp_ply)

            pc = o3d.io.read_point_cloud(tmp_ply)
            pts_raw = np.asarray(pc.points)

            # 这里只改清洗，不动 seed 和 inference 逻辑
            pts_clean = clean_point_cloud_np(pts_raw, voxel_size=0.008)

            if len(pts_clean) == 0:
                score = 1e9
                aux = None
                print(f"[SEED {seed}] empty after cleaning")
            else:
                if anchor_pts is not None and len(anchor_pts) > 0:
                    score, aux = coarse_cd_score_to_anchor(pts_clean, anchor_pts)
                    print(f"[SEED {seed}] raw_pts={len(pts_raw)} cleaned_pts={len(pts_clean)} rough_CD={score:.6f}")
                else:
                    score = -float(len(pts_clean))
                    aux = None
                    print(f"[SEED {seed}] raw_pts={len(pts_raw)} cleaned_pts={len(pts_clean)}")

            seed_logs.append({
                "seed": seed,
                "score": float(score),
                "num_points_raw": int(len(pts_raw)),
                "num_points_clean": int(len(pts_clean)),
                "aux": aux,
            })

            if keep_all_candidates:
                cand_dir = os.path.join(candidate_root, f"object_{oid:03d}")
                os.makedirs(cand_dir, exist_ok=True)
                cand_path = os.path.join(cand_dir, f"seed_{seed}.ply")
                cand_pc = make_point_cloud(pts_clean if len(pts_clean) > 0 else pts_raw)
                o3d.io.write_point_cloud(cand_path, cand_pc)

            if best is None or score < best["score"]:
                best = {
                    "seed": seed,
                    "score": float(score),
                    "points": pts_clean if len(pts_clean) > 0 else pts_raw,
                    "aux": aux,
                }

            if os.path.exists(tmp_ply):
                os.remove(tmp_ply)

        best_points = best["points"]
        output_filename = f"object_{oid:03d}.ply"
        output_path = os.path.join(output_dir, output_filename)

        best_pc = make_point_cloud(best_points)
        o3d.io.write_point_cloud(output_path, best_pc)

        print(f"[BEST] object_{oid:03d} seed={best['seed']} score={best['score']:.6f}")
        print(f"[SAVE] {output_path}")

        results.append({
            "object_id": oid,
            "mask_file": mf,
            "ply_file": output_filename,
            "ply_path": os.path.abspath(output_path),
            "best_seed": int(best["seed"]),
            "best_score": float(best["score"]),
            "best_aux": best["aux"],
            "seed_logs": seed_logs,
        })

    sam3d_manifest = {
        "sam3_manifest_path": os.path.abspath(sam3_manifest_path),
        "anchor_meta_path": os.path.abspath(anchor_meta_path) if anchor_meta_path else None,
        "image_path": os.path.abspath(image_path),
        "mask_dir": os.path.abspath(mask_dir),
        "temp_mask_dir": os.path.abspath(temp_mask_dir),
        "mask_files": mask_files,
        "num_seeds": int(num_seeds),
        "base_seed": int(base_seed),
        "results": results,
    }

    sam3d_manifest_path = os.path.join(output_dir, "sam3d_manifest.json")
    with open(sam3d_manifest_path, "w", encoding="utf-8") as f:
        json.dump(sam3d_manifest, f, indent=2, ensure_ascii=False)

    print("\n所有物体重建完成！")
    print(f"SAM3D manifest 已保存到: {sam3d_manifest_path}")
    return results


if __name__ == "__main__":
    reconstruct_multi_objects_bestofn(
        sam3_manifest_path="/home/wugui826/sam3/sam3_text_prompt_results/sam3_manifest.json",
        anchor_meta_path="/home/wugui826/vggt/anchors/meta.json",
        output_dir="/home/wugui826/sam-3d-objects/sam_ply",
        temp_mask_dir="/home/wugui826/sam3/_temp_sam3d_masks",
        num_seeds=2,
        base_seed=42,
        keep_all_candidates=False,
    )
