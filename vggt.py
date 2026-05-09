import os
import json
import shutil
import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from visual_util import predictions_to_glb
from demo_gradio import run_model


def load_sam3_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"SAM3 manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "items" not in manifest or len(manifest["items"]) == 0:
        raise ValueError(f"Invalid SAM3 manifest: no items in {manifest_path}")

    return manifest


def ensure_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim == 3:
        T = T[0]
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T
        return T4
    raise ValueError(f"Unsupported transform shape: {T.shape}")


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    shp = points.shape
    pts = points.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_t = (T @ pts_h.T).T[:, :3]
    return pts_t.reshape(shp)


def positive_depth_ratio(points_3d: np.ndarray) -> float:
    pts = points_3d.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0
    return float(np.mean(pts[:, 2] > 1e-6))


def derive_camera_points_from_world(predictions):
    if "world_points_from_depth" not in predictions:
        return None, None

    if "extrinsic" not in predictions or predictions["extrinsic"] is None:
        return None, None

    world_pts = predictions["world_points_from_depth"]
    T = ensure_4x4(predictions["extrinsic"])

    # 假设 1：extrinsic 是 world_to_cam
    cam_pts_a = transform_points(world_pts, T)
    score_a = positive_depth_ratio(cam_pts_a)

    # 假设 2：extrinsic 是 cam_to_world，需要取逆
    T_inv = np.linalg.inv(T)
    cam_pts_b = transform_points(world_pts, T_inv)
    score_b = positive_depth_ratio(cam_pts_b)

    if score_a >= score_b:
        return cam_pts_a, "camera_points_from_depth_derived_from_world_using_extrinsic"
    else:
        return cam_pts_b, "camera_points_from_depth_derived_from_world_using_inv_extrinsic"


def pick_or_build_camera_points(predictions):
    """
    最终目标：尽量返回 camera 坐标系点云
    """
    print("[DEBUG] prediction keys:", list(predictions.keys()))

    if "camera_points_from_depth" in predictions and predictions["camera_points_from_depth"] is not None:
        return predictions["camera_points_from_depth"], "camera_points_from_depth", "camera"

    if "points_from_depth" in predictions and predictions["points_from_depth"] is not None:
        # 很多实现里 points_from_depth 本身就是 camera frame
        return predictions["points_from_depth"], "points_from_depth", "camera"

    cam_pts, derived_name = derive_camera_points_from_world(predictions)
    if cam_pts is not None:
        return cam_pts, derived_name, "camera"

    if "world_points_from_depth" in predictions and predictions["world_points_from_depth"] is not None:
        return predictions["world_points_from_depth"], "world_points_from_depth", "world"

    raise KeyError(f"No usable point map found. Available keys: {list(predictions.keys())}")


def prepare_target_dir_from_manifest(manifest, work_target_dir="vggt_input"):
    os.makedirs(work_target_dir, exist_ok=True)
    image_subdir = os.path.join(work_target_dir, "images")
    os.makedirs(image_subdir, exist_ok=True)

    image_path = manifest.get("saved_image") or manifest.get("image_path")
    if image_path is None:
        raise ValueError("Manifest must contain 'saved_image' or 'image_path'")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image in manifest not found: {image_path}")

    dst_image_path = os.path.join(image_subdir, os.path.basename(image_path))

    if os.path.exists(dst_image_path):
        os.remove(dst_image_path)
    shutil.copy2(image_path, dst_image_path)

    return work_target_dir, dst_image_path


def extract_anchor_points_and_mask(point_map, mask_path, max_points=20000):
    pts = point_map
    if pts.ndim == 4:
        pts = pts[0]

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0)

    H, W = pts.shape[:2]
    if mask.shape[:2] != (H, W):
        mask_rs = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_rs = mask.astype(bool)

    obj_pts = pts[mask_rs]
    obj_pts = obj_pts[np.isfinite(obj_pts).all(axis=1)]

    if obj_pts.shape[0] > max_points:
        idx = np.random.choice(obj_pts.shape[0], max_points, replace=False)
        obj_pts = obj_pts[idx]

    return obj_pts, mask_rs, H, W


def main(
    sam3_manifest_path,
    out_anchor_dir="anchors",
    work_target_dir="vggt_input",
    export_glb=True,
    glb_path="result.glb",
    conf_thres=50,
    prediction_mode="Depthmap and Camera Branch",
):
    manifest = load_sam3_manifest(sam3_manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(sam3_manifest_path))

    print(f"[INFO] Loaded SAM3 manifest: {sam3_manifest_path}")

    target_dir, copied_image_path = prepare_target_dir_from_manifest(
        manifest,
        work_target_dir=work_target_dir
    )

    print(f"[INFO] VGGT target_dir prepared: {target_dir}")
    print(f"[INFO] Image copied to: {copied_image_path}")

    os.makedirs(out_anchor_dir, exist_ok=True)
    os.makedirs(os.path.join(out_anchor_dir, "masks_rs"), exist_ok=True)

    print("[INFO] Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    ))
    model = model.cuda().eval()
    print("[INFO] VGGT model loaded.")

    predictions = run_model(target_dir, model)

    point_map, point_key, coord_frame = pick_or_build_camera_points(predictions)
    print(f"[VGGT] use point key: {point_key}")
    print(f"[VGGT] coord_frame : {coord_frame}")

    H_out, W_out = None, None
    saved_items = []

    items = sorted(manifest["items"], key=lambda x: x["object_id"])
    print(f"[VGGT] Found {len(items)} masks in manifest")

    for item in items:
        idx = int(item["object_id"])
        mask_file = item["mask_file"]

        mask_path = os.path.join(manifest_dir, mask_file)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        pts, mask_rs, H, W = extract_anchor_points_and_mask(
            point_map,
            mask_path
        )

        if H_out is None:
            H_out, W_out = H, W

        anchor_filename = f"object_{idx:03d}.npy"
        mask_rs_filename = f"object_{idx:03d}.npy"

        np.save(os.path.join(out_anchor_dir, anchor_filename), pts)
        np.save(os.path.join(out_anchor_dir, "masks_rs", mask_rs_filename), mask_rs.astype(np.uint8))

        saved_items.append({
            "object_id": idx,
            "mask_file": mask_file,
            "anchor_file": anchor_filename,
            "mask_rs_file": f"masks_rs/{mask_rs_filename}",
            "num_points": int(pts.shape[0]),
        })

        print(f"[VGGT] anchor saved: {anchor_filename}  pts={pts.shape}")

    np.savez(
        os.path.join(out_anchor_dir, "scene_cam.npz"),
        intrinsic=predictions.get("intrinsic"),
        extrinsic=predictions.get("extrinsic"),
        depth=predictions.get("depth"),
        width=np.array([W_out], dtype=np.int32),
        height=np.array([H_out], dtype=np.int32),
    )
    print(f"[VGGT] scene_cam.npz saved to {out_anchor_dir}")

    meta = {
        "sam3_manifest_path": os.path.abspath(sam3_manifest_path),
        "source_image": manifest.get("saved_image") or manifest.get("image_path"),
        "point_key": point_key,
        "coord_frame": coord_frame,
        "height": int(H_out),
        "width": int(W_out),
        "items": saved_items,
    }

    meta_path = os.path.join(out_anchor_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[VGGT] meta saved: {meta_path}")

    if export_glb:
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames="All",
            show_cam=True,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        scene.export(glb_path)
        print(f"[VGGT] scene glb exported: {glb_path}")


if __name__ == "__main__":
    main(
        sam3_manifest_path="/home/wugui826/sam3/sam3_text_prompt_results/sam3_manifest.json",
        out_anchor_dir="/home/wugui826/vggt/anchors",
        work_target_dir="/home/wugui826/vggt/vggt_input",
        export_glb=True,
        glb_path="/home/wugui826/vggt/result.glb",
    )
