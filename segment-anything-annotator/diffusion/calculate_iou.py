from pathlib import Path
import argparse
import json
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

TASK_DICT = {
    "two_views_sphere": Path("./diffusion/data/eval_two_cameras_sphere"),
    "one_view_sphere": Path("./diffusion/data/eval_one_camera_sphere"),
    "two_views_cube": Path("./diffusion/temp/eval_two_camera_cube"),
    "one_view_cube": Path("./diffusion/data/eval_one_camera_cube"),
}

OUTPUT_DIR = "./diffusion/output"

def load_mask(img_shape, annotation_file):
    if not annotation_file.exists():
        mask = np.zeros_like(img_shape, dtype=np.uint8)
        return mask
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
        points = np.array(data["shapes"][0]["points"])
        
        hull = ConvexHull(data["shapes"][0]["points"])
        hull_points = points[hull.vertices].astype(np.int32)

        # 4. (Optional) Visualize the points and the convex hull
        mask = np.zeros(img_shape[:-1], dtype=np.uint8)
        cv2.fillPoly(mask, [hull_points], 1)

    return mask

def compute_iou(task_name, root_dir):
    anno_dir = root_dir / "annotations"
    img_dir = root_dir / "top_views"

    # Load target img
    target_file = anno_dir / "target.json"
    img_file = img_dir / (target_file.stem + ".png")
    img = cv2.imread(img_file.as_posix())
    target_mask = load_mask(img.shape, anno_dir / "target.json")

    # Get all img keys
    img_keys = [a.stem for a in img_dir.iterdir() if "target" not in a.stem]

    # Compute iou per image
    iou_array = []
    intersections = []
    for key in img_keys:
        if len(key) > 2 or int(key) > 31:
            continue

        anno_file = anno_dir / (key + ".json")
        img_file = img_dir / (key + ".jpg")
        if not anno_file.exists():
            iou_array.append(0.0)
            continue

        img = cv2.imread(img_file.as_posix())
        object_mask = load_mask(img.shape, anno_file)

        # caculate iou, normalize to the maximum iou found
        intersection = np.sum(np.logical_and(object_mask, target_mask))
        union = np.sum(np.logical_or(object_mask, target_mask))

        if union == 0:
            iou_array.append(0.0)
            continue
        
        iou = intersection / union
        iou_array.append(iou)
    
    iou_array = np.array(iou_array)
    normed_iou_array = iou_array / iou_array.max()
    mIou = normed_iou_array.sum() / len(normed_iou_array)

    print(f"{task_name} IOU: {mIou:.4f}")


def evaluate_task():
    for task_name, root_dir in TASK_DICT.items():
        compute_iou(task_name, root_dir)

def visualize():
    for task_name, root_dir in TASK_DICT.items():
        anno_dir = root_dir / "annotations"
        img_dir = root_dir / "top_views"

        # Load target img
        target_file = anno_dir / "target.json"
        img_file = img_dir / (target_file.stem + ".png")
        target_img = cv2.imread(img_file.as_posix())
        target_mask = load_mask(target_img.shape, anno_dir / "target.json").astype(bool)

        # Load random object img
        key = "22"
        anno_file = anno_dir / (key + ".json")
        object_img = cv2.imread(img_file.as_posix())
        object_mask = load_mask(object_img.shape, anno_file).astype(bool)

        # Color masks
        seg_img = np.zeros_like(target_img)
        seg_img[target_mask, 2] = 255
        seg_img[object_mask, :] = 255

        save_path = Path(OUTPUT_DIR) / f"{task_name}_{key}_segmentation.png"
        cv2.imwrite(save_path.as_posix(), seg_img)
        # cv2.imshow("segmentation", seg_img)
        # cv2.waitKey()


def main():
    parser = argparse.ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("--visualize", action="store_true", help="Evaluate two views sphere")
    args = parser.parse_args()

    if args.visualize:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        visualize()
    else:
        evaluate_task()

if __name__ == "__main__":
    main()