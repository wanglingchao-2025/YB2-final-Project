import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ===============================
# 工具函数：解码 SAM3 mask
# ===============================
def decode_sam3_mask(mask, image):
    """
    将 SAM3 输出的 mask 解码为 (H, W) 的 numpy array
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.array(mask)

    # 原图尺寸
    W, H = image.size

    # 去掉多余维度
    mask = np.squeeze(mask)

    if mask.ndim == 1:
        # (H*W,)
        assert mask.shape[0] == H * W, \
            f"Mask length {mask.shape[0]} != H*W ({H*W})"
        mask = mask.reshape(H, W)

    elif mask.ndim == 2:
        # (H, W)
        pass

    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    return mask


# ===============================
# 主函数：SAM3 文本提示分割并保存过滤后的结果
# ===============================
def run_sam3_text_prompt(
    image_path,
    text_prompt,
    save_dir="sam3_text_prompt_results",
    visualize=True,
    score_threshold=0.8,
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 模型加载
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 model loaded.")

    # 2. 加载图像
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    # 保存统一命名的原图
    saved_image_path = os.path.join(save_dir, "image.png")
    image.save(saved_image_path)

    inference_state = processor.set_image(image)

    # 3. 文本提示推理
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt
    )

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"\nText prompt: {text_prompt}")
    print(f"Original detected masks: {len(masks)}")
    print(f"Filtering with score > {score_threshold}\n")

    manifest_items = []

    # 保存筛选后的内容，供可视化使用
    filtered_masks = []
    filtered_boxes = []
    filtered_scores = []

    # 4. 仅保存 score > threshold 的 mask
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        score_value = float(score)

        if score_value <= score_threshold:
            print(f"[{i:02d}] score = {score_value:.4f} -> ignored")
            continue

        mask_2d = decode_sam3_mask(mask, image)

        # 二值化 -> uint8
        mask_img = (mask_2d > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_img, mode="L")

        save_idx = len(filtered_masks)
        mask_filename = f"mask_{save_idx:02d}.png"
        mask_path = os.path.join(save_dir, mask_filename)
        mask_pil.save(mask_path)

        if torch.is_tensor(box):
            box = box.detach().cpu().numpy()
        box = np.asarray(box).tolist()

        print(f"[{save_idx:02d}] score = {score_value:.4f}, saved to {mask_path}")
        print(f"     box = {box}")

        manifest_items.append({
            "object_id": save_idx,
            "mask_file": mask_filename,
            "score": score_value,
            "box": box,
        })

        filtered_masks.append(mask)
        filtered_boxes.append(box)
        filtered_scores.append(score_value)

    if len(filtered_masks) == 0:
        print("\nNo masks with score > threshold were found.")

    # 5. 保存 SAM3 manifest
    manifest = {
        "image_path": os.path.abspath(image_path),
        "saved_image": os.path.abspath(saved_image_path),
        "prompt": text_prompt,
        "image_width": W,
        "image_height": H,
        "score_threshold": score_threshold,
        "num_masks_before_filter": len(masks),
        "num_masks_after_filter": len(manifest_items),
        "items": manifest_items,
    }

    manifest_path = os.path.join(save_dir, "sam3_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSAM3 manifest saved to: {manifest_path}")

    # 6. 可视化
    if visualize:
        show_results(image, filtered_masks, filtered_boxes, filtered_scores, text_prompt, score_threshold)

    return manifest


# ===============================
# 可视化（可选）
# ===============================
def show_results(image, masks, boxes, scores, text_prompt, score_threshold=0.8):
    if len(masks) == 0:
        plt.figure(figsize=(6, 5))
        plt.imshow(image)
        plt.title(f'Original Image\nNo masks with score > {score_threshold}')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(4 * (len(masks) + 1), 5))

    # 当只有一个结果时，避免 axes 不是数组
    if len(masks) == 0:
        axes = [axes]
    elif len(masks) == 1:
        axes = np.array(axes).reshape(-1)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        axes[i + 1].imshow(image)

        mask_2d = decode_sam3_mask(mask, image)
        axes[i + 1].imshow(mask_2d, alpha=0.6, cmap="viridis")

        if torch.is_tensor(box):
            box = box.detach().cpu().numpy()

        x1, y1, x2, y2 = np.asarray(box).tolist()

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        axes[i + 1].add_patch(rect)
        axes[i + 1].set_title(f"mask_{i:02d}\nscore={float(score):.3f}")
        axes[i + 1].axis("off")

    plt.suptitle(f'Text Prompt: "{text_prompt}" | score > {score_threshold}', fontsize=14)
    plt.tight_layout()
    plt.show()


# ===============================
# 使用示例
# ===============================
if __name__ == "__main__":
    run_sam3_text_prompt(
        image_path="fruit.jpg",
        text_prompt="fruit",
        save_dir="sam3_text_prompt_results",
        visualize=True,
        score_threshold=0.8,
    )
