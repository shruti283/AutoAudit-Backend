import os
import cv2
import pytesseract
import pandas as pd

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(path):
    img = cv2.imread(path)
    if img is None:
        return ""
    text = pytesseract.image_to_string(img)
    return text

def generate_text_dataset(real_folder="data/real", fake_folder="data/fake", out_file="receipt_texts.csv"):
    rows = []

    for label, folder in [(1, real_folder), (0, fake_folder)]:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(("jpg", "jpeg", "png")):
                fpath = os.path.join(folder, fname)
                text = extract_text_from_image(fpath)

                rows.append([fname, text, label])
                print("Processed:", fname)

    df = pd.DataFrame(rows, columns=["filename", "text", "label"])
    df.to_csv(out_file, index=False)
    print("\n📁 Saved text dataset to:", out_file)

if __name__ == "__main__":
    generate_text_dataset()
