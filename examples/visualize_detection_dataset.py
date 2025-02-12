import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, r):
    x1, y1 = pt1
    x2, y2 = pt2

    thickness = max(1, min(thickness, cv2.LINE_AA))

    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Top right drawing
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_mask(image, mask, color, alpha=0.5):
    overlay = image.copy()
    if isinstance(mask, list):
        mask = np.array([[int(p[0]), int(p[1])] for p in mask])
        cv2.fillPoly(overlay, [mask], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    else:
        binary_mask = mask_utils.decode(mask)
        overlay[binary_mask == 1] = color
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_bboxes_and_labels(image, annotations, class_names):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 3
    bbox_thickness = 2
    text_color = (255, 255, 255)  # White text
    rectangle_radius = 8

    for i in range(len(annotations["boxes"])):
        bbox = annotations["boxes"][i]
        label = annotations["labels"][i]

        x_min, y_min, x_max, y_max = map(int, bbox)
        label_text = class_names[label]

        # Calculate text size and position
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = x_min
        text_y = y_min - 7
        background_top_left = (text_x, text_y - text_size[1])
        background_bottom_right = (text_x + text_size[0], text_y + 5)

        # Draw rounded rectangle
        draw_rounded_rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            bbox_thickness,
            rectangle_radius,
        )

        if "masks" in annotations:
            masks = annotations["masks"][i]
            # Choose random color
            color = np.random.randint(0, 256, size=3).tolist()
            draw_mask(image, masks, color, 0.5)

        # Draw text background
        draw_rounded_rectangle(
            image,
            background_top_left,
            background_bottom_right,
            (0, 255, 0),
            2,
            rectangle_radius,
        )

        # Put the text
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

    return image


def visualize_dataset(dataset_dir, save_images):
    annotations_path = os.path.join(dataset_dir, "annotations.json")

    with open(annotations_path) as file:
        all_annotations = json.load(file)

    class_names = all_annotations.pop("class_names", [])

    if save_images:
        if not os.path.exists(os.path.join(dataset_dir, "visualization")):
            os.makedirs(os.path.join(dataset_dir, "visualization"))
        vis_dir = os.path.join(dataset_dir, "visualization")

    for image_name, annotations in all_annotations.items():
        image_path = image_name
        image = cv2.imread(os.path.join(dataset_dir, image_path))
        if image is None:
            continue
        image = draw_bboxes_and_labels(image, annotations, class_names)

        if save_images:
            save_path = os.path.join(vis_dir, "bbox_" + os.path.basename(image_name))
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow("Image with Bounding Boxes", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes and labels on images."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the images with bounding boxes and labels instead of displaying",
    )
    args = parser.parse_args()

    visualize_dataset(args.dataset_path, args.save)


if __name__ == "__main__":
    main()
