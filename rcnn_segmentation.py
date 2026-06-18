from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import colorsys
import random
import time

import cv2
import matplotlib.pylab as plt
import numpy as np

DEFAULT_MAX_IMAGE_DIMENSION = 1200

@dataclass
class SegmentationResult:
    original_image: np.ndarray
    segmented_image: np.ndarray
    detections: list[dict]
    num_classes: int
    num_detections: int
    elapsed_time: float
    resized: bool = False
    original_size: Optional[Tuple[int, int]] = None
    processed_size: Optional[Tuple[int, int]] = None

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda color: colorsys.hsv_to_rgb(*color), hsv))
    random.shuffle(colors)
    return 256 * np.array(colors)

def get_model_files(model_path="models"):
    model_dir = Path(model_path)

    labels_path = model_dir / "object_detection_classes_coco.txt"
    weights_path = model_dir / "frozen_inference_graph.pb"
    config_path = model_dir / "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    return labels_path, weights_path, config_path

def load_labels(labels_path):
    if not labels_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {labels_path}")

    return labels_path.read_text(encoding="utf-8").strip().split("\n")

def load_model(weights_path, config_path):
    if not weights_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {weights_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    return cv2.dnn.readNetFromTensorflow(str(weights_path), str(config_path))

def load_default_model(model_path="models"):
    labels_path, weights_path, config_path = get_model_files(model_path)
    labels = load_labels(labels_path)
    net = load_model(weights_path, config_path)
    return net, labels

def read_image_from_path(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibaca: {image_path}")

    return image

def read_image_from_bytes(file_bytes):
    image_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("File upload tidak bisa dibaca sebagai gambar.")

    return image

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_for_inference(image, max_dimension=DEFAULT_MAX_IMAGE_DIMENSION):
    h, w = image.shape[:2]
    largest_dimension = max(h, w)

    if largest_dimension <= max_dimension:
        return image, False, (w, h), (w, h)

    scale = max_dimension / largest_dimension
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized_image, True, (w, h), (new_w, new_h)

def get_label(labels, class_id):
    if class_id < len(labels):
        return labels[class_id]

    shifted_class_id = class_id - 1

    if 0 <= shifted_class_id < len(labels):
        return labels[shifted_class_id]

    return f"class_{class_id}"

def run_instance_segmentation(
    image,
    net,
    labels,
    conf=0.5,
    thresh=0.3,
    max_dimension=DEFAULT_MAX_IMAGE_DIMENSION
):
    image, resized, original_size, processed_size = resize_for_inference(
        image,
        max_dimension=max_dimension
    )

    original_image = image.copy()
    segmented_image = image.copy()

    blob = cv2.dnn.blobFromImage(segmented_image, swapRB=True, crop=False)

    net.setInput(blob)

    start = time.time()

    boxes, masks = net.forward([
        "detection_out_final",
        "detection_masks"
    ])

    elapsed_time = time.time() - start

    num_classes = masks.shape[1]
    num_detections = boxes.shape[2]

    print("# instances: {}".format(num_detections))
    print("# classes: {}".format(num_classes))

    colors = random_colors(num_detections)

    h = segmented_image.shape[0]
    w = segmented_image.shape[1]

    detections = []

    for i in range(num_detections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]

        if score > conf:
            class_id = int(box[1])

            if class_id >= num_classes:
                continue

            left, top, right, bottom = int(w * box[3]), int(h * box[4]), \
                int(w * box[5]), int(h * box[6])

            left, top = max(0, min(left, w - 1)), max(0, min(top, h - 1))
            right, bottom = max(0, min(right, w - 1)), \
                max(0, min(bottom, h - 1))

            if right <= left or bottom <= top:
                continue

            class_mask = mask[class_id]
            label = get_label(labels, class_id)

            class_mask = cv2.resize(
                class_mask,
                (right - left + 1, bottom - top + 1)
            )

            mask = class_mask > thresh
            roi = segmented_image[top:bottom + 1, left:right + 1][mask]

            if roi.size == 0:
                continue

            color_index = np.random.randint(0, len(colors)) if len(colors) > 0 else 0
            color = colors[color_index].astype(np.uint8)
            color_tuple = tuple(int(value) for value in color)

            segmented_image[top:bottom + 1, left:right + 1][mask] = (
                0.4 * color + 0.6 * roi
            ).astype(np.uint8)

            mask = mask.astype(np.uint8)

            contours, hierarchy = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cv2.drawContours(
                    segmented_image[top:bottom + 1, left:right + 1],
                    contours,
                    -1,
                    color_tuple,
                    3,
                    cv2.LINE_8,
                    hierarchy,
                    100
                )

            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            label_top = max(top, label_size[1])

            cv2.putText(
                segmented_image,
                label,
                ((left + right) // 2, label_top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2
            )

            detections.append({
                "label": label,
                "confidence": round(float(score), 4),
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom
            })

            print(label, "{:.2f}".format(score))

    return SegmentationResult(
        original_image=original_image,
        segmented_image=segmented_image,
        detections=detections,
        num_classes=int(num_classes),
        num_detections=int(num_detections),
        elapsed_time=float(elapsed_time),
        resized=resized,
        original_size=original_size,
        processed_size=processed_size
    )

def show_comparison(original_image, segmented_image):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(bgr_to_rgb(original_image))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(bgr_to_rgb(segmented_image))
    plt.title("Instance segmentation (Mask R-CNN)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    net, labels = load_default_model("models")
    image = read_image_from_path("images/image_placeholder.png")

    result = run_instance_segmentation(
        image=image,
        net=net,
        labels=labels,
        conf=0.5,
        thresh=0.3
    )

    print("Waktu forward pass: {:.2f} detik".format(result.elapsed_time))

    show_comparison(
        result.original_image,
        result.segmented_image
    )

main()