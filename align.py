import os
import json
import itertools
import numpy as np
import cv2
import open3d as o3d


# =========================================================
# 0) IO
# =========================================================
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mask_rs_bool(mask_rs_path: str) -> np.ndarray:
    if not os.path.exists(mask_rs_path):
        raise FileNotFoundError(mask_rs_path)
    m = np.load(mask_rs_path)
    return (m > 0)


def build_records(sam3d_manifest_path, anchor_meta_path):
    sam3d = load_json(sam3d_manifest_path)
    meta = load_json(anchor_meta_path)

    sam3d_dir = os.path.dirname(os.path.abspath(sam3d_manifest_path))
    anchor_dir = os.path.dirname(os.path.abspath(anchor_meta_path))

    records = []
    sam3d_map = {int(x["object_id"]): x for x in sam3d["results"]}
    anchor_map = {int(x["object_id"]): x for x in meta["items"]}

    common_ids = sorted(set(sam3d_map.keys()) & set(anchor_map.keys()))
    if len(common_ids) == 0:
        raise ValueError("No common object_id")

    for oid in common_ids:
        s3d = sam3d_map[oid]
        am = anchor_map[oid]

        ply_path = s3d.get("ply_path")
        if ply_path is None:
            ply_file = s3d.get("ply_file")
            if ply_file is None:
                raise ValueError(f"sam3d_manifest missing ply for object_id={oid}")
            ply_path = os.path.join(sam3d_dir, ply_file)

        anchor_path = os.path.join(anchor_dir, am["anchor_file"])
        mask_rs_path = os.path.join(anchor_dir, am["mask_rs_file"])

        if not os.path.exists(ply_path):
            raise FileNotFoundError(ply_path)
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(anchor_path)
        if not os.path.exists(mask_rs_path):
            raise FileNotFoundError(mask_rs_path)

        records.append({
            "object_id": oid,
            "ply_path": ply_path,
            "anchor_path": anchor_path,
            "mask_rs_path": mask_rs_path,
        })

    print(f"[INFO] Loaded {len(records)} records")
    print(f"[INFO] coord_frame : {meta.get('coord_frame', 'unknown')}")
    print(f"[INFO] point_key   : {meta.get('point_key', 'unknown')}")

    return {
        "records": records,
        "coord_frame": meta.get("coord_frame", "world"),
        "height": int(meta["height"]),
        "width": int(meta["width"]),
    }


# =========================================================
# 1) Basic utils
# =========================================================
def make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pc


def voxel_downsample_np(points, voxel_size):
    if len(points) == 0:
        return points
    pc = make_point_cloud(points)
    pc = pc.voxel_down_sample(voxel_size)
    return np.asarray(pc.points)


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


def fscore_from_points(pred_pts, gt_pts, threshold=0.05):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    pc_pred = make_point_cloud(pred_pts)
    pc_gt = make_point_cloud(gt_pts)

    d1 = np.asarray(pc_pred.compute_point_cloud_distance(pc_gt))
    d2 = np.asarray(pc_gt.compute_point_cloud_distance(pc_pred))

    precision = np.mean(d1 < threshold) if len(d1) > 0 else 0.0
    recall = np.mean(d2 < threshold) if len(d2) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def clean_point_cloud_np(points, voxel_size=0.01, use_dbscan=True):
    if len(points) == 0:
        return points

    pc = make_point_cloud(points)
    pc = pc.voxel_down_sample(voxel_size)

    if len(pc.points) >= 20:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pc.points) >= 20:
        pc, _ = pc.remove_radius_outlier(nb_points=12, radius=voxel_size * 4)

    if use_dbscan and len(pc.points) >= 30:
        labels = np.array(pc.cluster_dbscan(eps=voxel_size * 6, min_points=20, print_progress=False))
        if len(labels) > 0 and np.max(labels) >= 0:
            major = np.argmax(np.bincount(labels[labels >= 0]))
            pts = np.asarray(pc.points)[labels == major]
        else:
            pts = np.asarray(pc.points)
    else:
        pts = np.asarray(pc.points)

    return pts


def bbox_center_extent(points):
    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    center = 0.5 * (mn + mx)
    extent = mx - mn
    return center, extent


def apply_axis_transform(points: np.ndarray, perm=(0, 1, 2), sign=(1, 1, 1)) -> np.ndarray:
    pts = points[:, perm].copy()
    pts *= np.array(sign, dtype=np.float64)[None, :]
    return pts


def yaw_rotation(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def apply_transform_local_to_camera(points_local, scale, trans, perm=(0, 1, 2), sign=(1, 1, 1), yaw_deg=0.0):
    pts = apply_axis_transform(points_local, perm=perm, sign=sign)
    R = yaw_rotation(yaw_deg)
    pts = pts @ R.T
    pts = scale * pts + trans[None, :]
    return pts


def scale_intrinsic(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def subsample_points_np(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx]


# =========================================================
# 2) Camera
# =========================================================
def load_camera_for_maskrs(scene_cam_path, H_mask, W_mask):
    cam = np.load(scene_cam_path, allow_pickle=True)

    K = cam["intrinsic"]
    if K.ndim == 3:
        K = K[0]
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]

    if "width" not in cam or "height" not in cam:
        raise ValueError("scene_cam.npz must contain width and height")

    W0 = int(np.asarray(cam["width"]).reshape(-1)[0])
    H0 = int(np.asarray(cam["height"]).reshape(-1)[0])

    sx = W_mask / max(W0, 1e-9)
    sy = H_mask / max(H0, 1e-9)
    K_scaled = scale_intrinsic(K, sx, sy)

    return K_scaled


# =========================================================
# 3) Projection / mask rendering
# =========================================================
def project_camera_points(K, X):
    if len(X) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)

    valid = X[:, 2] > 1e-6
    Xv = X[valid]
    if len(Xv) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)

    idx = np.where(valid)[0]
    z = Xv[:, 2]
    x = Xv[:, 0] / z
    y = Xv[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, z, idx


def render_visible_mask_fast(
    uv,
    depth,
    H,
    W,
    point_radius=4,
    dilate_iter=2,
    close_iter=2,
    use_convex_hull=True,
):
    mask = np.zeros((H, W), dtype=np.uint8)
    visible_keep = np.zeros((uv.shape[0],), dtype=bool)

    if uv.shape[0] == 0:
        return mask.astype(bool), visible_keep

    uv_i = np.round(uv).astype(np.int32)
    valid = (
        (uv_i[:, 0] >= 0) & (uv_i[:, 0] < W) &
        (uv_i[:, 1] >= 0) & (uv_i[:, 1] < H)
    )
    if not np.any(valid):
        return mask.astype(bool), visible_keep

    uv_i = uv_i[valid]
    depth_v = depth[valid]
    orig_idx = np.where(valid)[0]

    xs = uv_i[:, 0]
    ys = uv_i[:, 1]
    lin = ys.astype(np.int64) * W + xs.astype(np.int64)

    order = np.lexsort((depth_v, lin))
    lin_sorted = lin[order]
    keep_first = np.ones(len(order), dtype=bool)
    keep_first[1:] = lin_sorted[1:] != lin_sorted[:-1]
    keep = order[keep_first]

    xs = xs[keep]
    ys = ys[keep]
    visible_keep[orig_idx[keep]] = True

    mask[ys, xs] = 255

    if point_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * point_radius + 1, 2 * point_radius + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    if use_convex_hull and len(xs) >= 3:
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        hull_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        mask = np.maximum(mask, hull_mask)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return mask.astype(bool), visible_keep


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-9
    return float(inter / union)


# =========================================================
# 4) Placement search
# =========================================================
def make_search_levels(base_scale, diag):
    coarse = {
        "yaw_candidates": list(range(-180, 181, 45)),
        "scale_muls": [0.80, 1.00, 1.20],
        "offset_fracs_xy": [-0.10, 0.0, 0.10],
        "offset_fracs_z": [-0.08, 0.0, 0.08],
        "topk": 10,
        "point_radius": 3,
        "dilate_iter": 1,
        "close_iter": 1,
        "use_convex_hull": True,
    }
    fine = {
        "yaw_delta": [-15, 0, 15],
        "scale_muls": [0.92, 1.00, 1.08],
        "offset_fracs_xy": [-0.04, 0.0, 0.04],
        "offset_fracs_z": [-0.03, 0.0, 0.03],
        "point_radius": 4,
        "dilate_iter": 2,
        "close_iter": 2,
        "use_convex_hull": True,
    }
    return coarse, fine


def score_candidate(local_centered, anchor_center, gt_mask, K, diag, cand, render_cfg):
    X = apply_transform_local_to_camera(
        local_centered,
        scale=cand["scale"],
        trans=cand["trans"],
        perm=cand["perm"],
        sign=cand["sign"],
        yaw_deg=cand["yaw_deg"],
    )
    uv, depth, _ = project_camera_points(K, X)
    pred_mask, visible_keep = render_visible_mask_fast(
        uv, depth, gt_mask.shape[0], gt_mask.shape[1],
        point_radius=render_cfg["point_radius"],
        dilate_iter=render_cfg["dilate_iter"],
        close_iter=render_cfg["close_iter"],
        use_convex_hull=render_cfg["use_convex_hull"],
    )
    iou = mask_iou(pred_mask, gt_mask)
    offset_penalty = np.linalg.norm(cand["trans"] - anchor_center) / max(diag, 1e-9)
    scale_penalty = abs(cand["scale"] / max(cand["base_scale"], 1e-9) - 1.0)
    score = iou - 0.05 * offset_penalty - 0.02 * scale_penalty
    return {
        "score": float(score),
        "iou": float(iou),
        "visible_keep": visible_keep,
        "pred_mask": pred_mask,
    }


def place_one_object_optimized(local_pts, anchor_pts, gt_mask, K, search_points_limit=3000):
    local_center, local_extent = bbox_center_extent(local_pts)
    local_centered_full = local_pts - local_center[None, :]
    local_centered_search = subsample_points_np(local_centered_full, search_points_limit)

    anchor_center, anchor_extent = bbox_center_extent(anchor_pts)
    base_scale = float(np.median((anchor_extent + 1e-9) / (local_extent + 1e-9)))
    diag = float(np.linalg.norm(anchor_extent) + 1e-9)

    perms = list(itertools.permutations([0, 1, 2]))
    signs = list(itertools.product([1, -1], repeat=3))
    coarse_cfg, fine_cfg = make_search_levels(base_scale, diag)

    coarse_results = []
    for perm in perms:
        for sign in signs:
            for yaw_deg in coarse_cfg["yaw_candidates"]:
                for sm in coarse_cfg["scale_muls"]:
                    scale = base_scale * sm
                    for fx in coarse_cfg["offset_fracs_xy"]:
                        for fy in coarse_cfg["offset_fracs_xy"]:
                            for fz in coarse_cfg["offset_fracs_z"]:
                                cand = {
                                    "perm": perm,
                                    "sign": sign,
                                    "yaw_deg": float(yaw_deg),
                                    "scale": float(scale),
                                    "base_scale": float(base_scale),
                                    "trans": anchor_center + diag * np.array([fx, fy, fz], dtype=np.float64),
                                }
                                res = score_candidate(
                                    local_centered_search,
                                    anchor_center,
                                    gt_mask,
                                    K,
                                    diag,
                                    cand,
                                    coarse_cfg,
                                )
                                cand.update(res)
                                coarse_results.append(cand)

    coarse_results.sort(key=lambda x: x["score"], reverse=True)
    coarse_results = coarse_results[:coarse_cfg["topk"]]

    best = None
    for coarse in coarse_results:
        base_trans = coarse["trans"]
        for yaw_delta in fine_cfg["yaw_delta"]:
            yaw_deg = coarse["yaw_deg"] + yaw_delta
            for sm in fine_cfg["scale_muls"]:
                scale = coarse["scale"] * sm
                for fx in fine_cfg["offset_fracs_xy"]:
                    for fy in fine_cfg["offset_fracs_xy"]:
                        for fz in fine_cfg["offset_fracs_z"]:
                            cand = {
                                "perm": coarse["perm"],
                                "sign": coarse["sign"],
                                "yaw_deg": float(yaw_deg),
                                "scale": float(scale),
                                "base_scale": float(base_scale),
                                "trans": base_trans + diag * np.array([fx, fy, fz], dtype=np.float64),
                            }
                            res = score_candidate(
                                local_centered_search,
                                anchor_center,
                                gt_mask,
                                K,
                                diag,
                                cand,
                                fine_cfg,
                            )
                            cand.update(res)
                            if best is None or cand["score"] > best["score"]:
                                best = cand

    X_final = apply_transform_local_to_camera(
        local_centered_full,
        scale=best["scale"],
        trans=best["trans"],
        perm=best["perm"],
        sign=best["sign"],
        yaw_deg=best["yaw_deg"],
    )
    best["points_cam"] = X_final
    return best


# =========================================================
# 5) Metrics / evaluation
# =========================================================
def evaluate_single_object(pred_pts_cam, anchor_pts, gt_mask, K, voxel_size=0.02, fscore_threshold=0.05):
    H, W = gt_mask.shape

    pred_mask, _ = render_visible_mask_fast(
        *project_camera_points(K, pred_pts_cam)[:2],
        H, W,
        point_radius=4,
        dilate_iter=2,
        close_iter=2,
        use_convex_hull=True,
    )
    iou = mask_iou(pred_mask, gt_mask)

    pred_ds = voxel_downsample_np(pred_pts_cam, voxel_size)
    anchor_ds = voxel_downsample_np(anchor_pts, voxel_size)

    cd = robust_chamfer_distance_np(pred_ds, anchor_ds, trim_ratio=0.90)
    fs = fscore_from_points(pred_ds, anchor_ds, threshold=fscore_threshold)

    return {
        "iou": float(iou),
        "cd": float(cd),
        "fscore": float(fs),
    }


def prepare_records(records, clean_voxel_size=0.01, use_dbscan=True):
    prepared = []
    for rec in records:
        local_raw = np.asarray(o3d.io.read_point_cloud(rec["ply_path"]).points)
        anchor_raw = np.asarray(np.load(rec["anchor_path"]), dtype=np.float64)
        gt_mask = load_mask_rs_bool(rec["mask_rs_path"])

        local_pts = clean_point_cloud_np(local_raw, voxel_size=clean_voxel_size, use_dbscan=use_dbscan)
        anchor_pts = clean_point_cloud_np(anchor_raw, voxel_size=clean_voxel_size, use_dbscan=use_dbscan)

        if len(local_pts) == 0:
            raise ValueError(f"Empty SAM3D point cloud after cleaning: {rec['ply_path']}")
        if len(anchor_pts) == 0:
            raise ValueError(f"Empty anchor after cleaning: {rec['anchor_path']}")

        item = dict(rec)
        item["local_pts"] = local_pts
        item["anchor_pts"] = anchor_pts
        item["gt_mask"] = gt_mask
        prepared.append(item)
    return prepared


def evaluate_composed_scene_prepared(prepared_records, aligned_points_map, scene_cam_path, voxel_size=0.02, fscore_threshold=0.05):
    print("\n========== Evaluation ==========")

    H_mask, W_mask = prepared_records[0]["gt_mask"].shape
    K = load_camera_for_maskrs(scene_cam_path, H_mask, W_mask)

    all_iou, all_cd, all_f = [], [], []
    per_object_metrics = []

    for rec in prepared_records:
        oid = rec["object_id"]
        pred_pts = aligned_points_map[oid]
        anchor_pts = rec["anchor_pts"]
        gt_mask = rec["gt_mask"]

        metrics = evaluate_single_object(
            pred_pts_cam=pred_pts,
            anchor_pts=anchor_pts,
            gt_mask=gt_mask,
            K=K,
            voxel_size=voxel_size,
            fscore_threshold=fscore_threshold,
        )

        per_object_metrics.append({
            "object_id": int(oid),
            "iou": metrics["iou"],
            "cd": metrics["cd"],
            "fscore": metrics["fscore"],
        })

        all_iou.append(metrics["iou"])
        all_cd.append(metrics["cd"])
        all_f.append(metrics["fscore"])

        print(
            f"[EVAL object_{oid:03d}] "
            f"IoU={metrics['iou']:.3f}  "
            f"CD={metrics['cd']:.5f}  "
            f"F-score={metrics['fscore']:.3f}"
        )

    summary = {
        "mean_iou": float(np.mean(all_iou)) if len(all_iou) > 0 else 0.0,
        "mean_cd": float(np.mean(all_cd)) if len(all_cd) > 0 else 0.0,
        "mean_fscore": float(np.mean(all_f)) if len(all_f) > 0 else 0.0,
    }

    print("\n===== Overall =====")
    print(f"Mean IoU     : {summary['mean_iou']:.4f}")
    print(f"Mean CD      : {summary['mean_cd']:.6f}")
    print(f"Mean F-score : {summary['mean_fscore']:.4f}")

    return {
        "per_object": per_object_metrics,
        "summary": summary,
    }


# =========================================================
# 6) Scene composition
# =========================================================
def compose_scene_optimized(
    prepared_records,
    scene_cam_path,
    out_dir,
    voxel_size=0.02,
    fscore_threshold=0.05,
    search_points_limit=3000,
):
    os.makedirs(out_dir, exist_ok=True)

    H_mask, W_mask = prepared_records[0]["gt_mask"].shape
    K = load_camera_for_maskrs(scene_cam_path, H_mask, W_mask)

    aligned_paths = []
    aligned_points_map = {}
    merged = o3d.geometry.PointCloud()
    placement_logs = []

    print("\n========== Compose Scene ==========")

    for rec in prepared_records:
        oid = rec["object_id"]
        local_pts = rec["local_pts"]
        anchor_pts = rec["anchor_pts"]
        gt_mask = rec["gt_mask"]

        info = place_one_object_optimized(
            local_pts=local_pts,
            anchor_pts=anchor_pts,
            gt_mask=gt_mask,
            K=K,
            search_points_limit=search_points_limit,
        )
        pts_cam = info["points_cam"]

        pc = make_point_cloud(pts_cam)
        out_path = os.path.join(out_dir, f"object_{oid:03d}_aligned.ply")
        o3d.io.write_point_cloud(out_path, pc)

        aligned_paths.append(out_path)
        aligned_points_map[oid] = pts_cam
        merged += pc

        placement_logs.append({
            "object_id": int(oid),
            "perm": list(info["perm"]),
            "sign": list(info["sign"]),
            "yaw_deg": float(info["yaw_deg"]),
            "placement_score": float(info["score"]),
            "placement_iou": float(info["iou"]),
        })

        print(
            f"[OK] object_{oid:03d} "
            f"perm={info['perm']} sign={info['sign']} yaw={info['yaw_deg']:+.1f} "
            f"placement_IoU={info['iou']:.3f}"
        )

    merged_path = os.path.join(out_dir, "merged_aligned_scene.ply")
    o3d.io.write_point_cloud(merged_path, merged)
    print(f"[MERGE] saved merged point cloud: {merged_path}")

    metrics = evaluate_composed_scene_prepared(
        prepared_records=prepared_records,
        aligned_points_map=aligned_points_map,
        scene_cam_path=scene_cam_path,
        voxel_size=voxel_size,
        fscore_threshold=fscore_threshold,
    )

    return aligned_paths, aligned_points_map, merged_path, placement_logs, metrics


# =========================================================
# 7) Main
# =========================================================
if __name__ == "__main__":
    SAM3D_MANIFEST_PATH = "/home/wugui826/sam-3d-objects/sam_ply/sam3d_manifest.json"
    ANCHOR_META_PATH = "/home/wugui826/vggt/anchors/meta.json"
    SCENE_CAM_PATH = "/home/wugui826/vggt/anchors/scene_cam.npz"
    OUT_DIR = "/home/wugui826/sam-3d-objects/aligned"

    VOXEL_SIZE = 0.02
    FSCORE_THRESHOLD = 0.05
    CLEAN_VOXEL_SIZE = 0.01
    SEARCH_POINTS_LIMIT = 3000
    USE_DBSCAN = True

    ctx = build_records(
        sam3d_manifest_path=SAM3D_MANIFEST_PATH,
        anchor_meta_path=ANCHOR_META_PATH,
    )

    if ctx["coord_frame"] != "camera":
        raise ValueError(f"Expected coord_frame='camera', got: {ctx['coord_frame']}")

    prepared_records = prepare_records(
        records=ctx["records"],
        clean_voxel_size=CLEAN_VOXEL_SIZE,
        use_dbscan=USE_DBSCAN,
    )

    aligned_paths, aligned_points_map, merged_path, placement_logs, metrics = compose_scene_optimized(
        prepared_records=prepared_records,
        scene_cam_path=SCENE_CAM_PATH,
        out_dir=OUT_DIR,
        voxel_size=VOXEL_SIZE,
        fscore_threshold=FSCORE_THRESHOLD,
        search_points_limit=SEARCH_POINTS_LIMIT,
    )

    metrics_out = {
        "placement_logs": placement_logs,
        "evaluation": metrics,
        "merged_scene_path": merged_path,
    }

    metrics_json_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] metrics json saved to: {metrics_json_path}")
