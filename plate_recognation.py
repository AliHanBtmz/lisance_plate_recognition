import cv2
import os


def load_templates(template_folder="plate"):
    templates = {}
    for fname in os.listdir(template_folder):
        if fname.lower().endswith(".bmp"):
            char = os.path.splitext(fname)[0].upper()
            path = os.path.join(template_folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (24, 42))
            templates[char] = img
    return templates


def match_template(char_img, templates):
    max_score = -1
    best_match = ""
    char_img = cv2.resize(char_img, (24, 42))
    for char, tmpl in templates.items():
        res = cv2.matchTemplate(char_img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        if score > max_score:
            max_score = score
            best_match = char
    return best_match if max_score > 0.4 else ""  # eşik değeri ayarlanabilir
