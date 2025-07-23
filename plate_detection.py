import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
from plate_recognation import load_templates, match_template

folder_path = "cars/"
image_files = sorted(os.listdir(folder_path))
image_index = 0
step_index = 0


def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(gray_img)
    return clahe_img


def show_step(title, image):
    temp = image.copy()
    cv2.putText(temp, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow("Step-by-Step Image Processing", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(
    # "Step-by-Step Image Processing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    # )

    cv2.imshow("Step-by-Step Image Processing", temp)
    cv2.moveWindow("Step-by-Step Image Processing", 550, 150)


def process_image_steps(img):
    steps = []
    templates = load_templates("plate")

    img = cv2.resize(img, (640, 480))
    steps.append(("Resized", img.copy()))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 35
    steps.append(("Grayscale", gray.copy()))  # 43

    median_val = np.median(gray)
    low = 0.67 * median_val
    high = 1.33 * median_val

    edges = cv2.Canny(gray, low, high)
    steps.append(("Canny Edges", edges.copy()))

    dilated = cv2.dilate(edges, np.ones((1, 2), np.uint8), iterations=1)
    steps.append(("Dilated Edges", dilated.copy()))

    skeleton = skeletonize(dilated)
    skeleton = (skeleton * 255).astype(np.uint8)
    steps.append(("Skeletonized", skeleton.copy()))

    clone = img.copy()
    contours = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    all_contours_img = img.copy()
    cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 1)
    steps.append(("All Contours", all_contours_img.copy()))

    gray_cropped = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    detected_centers = []

    roi_counter = 0

    candidate_boxes_img = clone.copy()
    for c in contours:
        rect = cv2.minAreaRect(c)
        (x, y), (h, w), angle = rect
        box_area = h * w
        if box_area > 10000:
            continue

        angle_deg = (angle + 90) % 180
        if w == max(h, w):
            angle_deg += 90

        if (
            (w > h > 8 and h * 5 > w > h * 2.5) or (h > w > 8 and w * 5 > h > w * 2.5)
        ) and ((265 < angle_deg < 300) or (70 < angle_deg < 94)):
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(candidate_boxes_img, [box], 0, (255, 0, 0), 2)
    steps.append(("Candidate Boxes", candidate_boxes_img.copy()))

    for c in contours:
        rect = cv2.minAreaRect(c)
        (x, y), (h, w), angle = rect
        box_area = h * w
        if box_area > 10000:
            continue

        angle_deg = (angle + 90) % 180
        if w == max(h, w):
            angle_deg += 90

        if (
            (w > h > 8 and h * 5 > w > h * 2.5) or (h > w > 8 and w * 5 > h > w * 2.5)
        ) and ((265 < angle_deg < 300) or (70 < angle_deg < 94)):
            box = cv2.boxPoints(rect).astype(np.int32)
            minx, miny = np.min(box[:, 0]), np.min(box[:, 1])
            maxx, maxy = np.max(box[:, 0]), np.max(box[:, 1])

            roi = gray_cropped[miny:maxy, minx:maxx]

            if roi is None or roi.size == 0:
                continue
            # ROI küçükse büyüt

            roi_height, roi_width = roi.shape[:2]

            scale_factor = 2.85 if roi_height < 15 or roi_width < 43 else 1.0
            roi = cv2.resize(
                roi,
                (int(roi_width * scale_factor), int(roi_height * scale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )

            roi_counter += 1
            steps.append(("", roi.copy()))

            hist = apply_clahe(roi)
            steps.append(("", hist.copy()))

            thresh_1 = cv2.adaptiveThreshold(
                hist,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                10,
            )
            steps.append(("", thresh_1.copy()))

            _, _, stats, _ = cv2.connectedComponentsWithStats(thresh_1)

            char_imgs = []
            for stat in stats[1:]:
                x1, y1, w1, h1, area = stat
                if 5 < area < 500 and h1 > 10 and w1 > 2:
                    char = thresh_1[y1 : y1 + h1, x1 : x1 + w1]
                    char_resized = cv2.resize(char, (24, 42))
                    padded = cv2.copyMakeBorder(
                        char_resized, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0
                    )
                    char_imgs.append((x1, padded))

            char_imgs.sort(key=lambda x: x[0])
            plate_text = "".join(
                [match_template(img, templates) for _, img in char_imgs]
            )

            if len(plate_text.strip()) < 4:
                continue

            center = (int(x), int(y))
            too_close = any(
                np.linalg.norm(np.array(center) - np.array(prev)) < 30
                for prev in detected_centers
            )
            if too_close:
                continue

            detected_centers.append(center)

            for i, (_, img) in enumerate(char_imgs):
                steps.append((f"{i+1}", img.copy()))

            color = (0, 255, 0)
            cv2.drawContours(clone, [box], 0, color, 2)
            cv2.putText(
                clone,
                plate_text,
                (minx, miny - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    steps.append(("Plate Detection", clone.copy()))
    return steps


def load_image(index):
    img_path = os.path.join(folder_path, image_files[index])
    img = cv2.imread(img_path)
    return process_image_steps(img)


steps = load_image(image_index)

while True:
    title, frame = steps[step_index]
    show_step(title, frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):  # ESC
        break
    elif key == ord("d"):  # next step
        step_index = (step_index + 1) % len(steps)
    elif key == ord("a"):  # previous step
        step_index = (step_index - 1) % len(steps)
    elif key == ord("c"):  # next image
        image_index = (image_index + 1) % len(image_files)
        steps = load_image(image_index)
        step_index = 0
    elif key == ord("z"):  # previous image
        image_index = (image_index - 1) % len(image_files)
        steps = load_image(image_index)
        step_index = 0
    elif key == ord("s"):  # tüm adımları kaydet
        save_dir = "output"
        os.makedirs(save_dir, exist_ok=True)
        for i, (step_title, img) in enumerate(steps):
            safe_title = step_title if step_title else f"step_{i}"
            safe_title = safe_title.replace(" ", "_").replace(":", "")
            filename = f"{image_index:02}_{i:02}_{safe_title}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img)
        print(f"Saved all steps for image {image_index + 1} to '{save_dir}' folder.")


cv2.destroyAllWindows()
