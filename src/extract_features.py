# src/extract_features.py
import os
import re
import cv2
import pytesseract
import pandas as pd

# UPDATE this path if Tesseract is in a different location on your machine:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

NUM_RE = re.compile(r"\d{1,3}(?:[,]\d{3})*(?:\.\d{1,2})?|\d+\.\d{1,2}")

def parse_amounts(text):
    # find monetary patterns like 12.50 or 1,234.56
    raw = NUM_RE.findall(text)
    # normalize commas
    amounts = []
    for r in raw:
        r2 = r.replace(",", "")
        try:
            v = float(r2)
            amounts.append(v)
        except:
            continue
    return amounts

def extract_bill_features(image_path, label=None):
    """
    Returns a dict of features for a single image path.
    label: optional (1 for real, -1 for fake) used when building dataset
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read {image_path}")
        return None

    # basic preprocessing (grayscale + slight blur can help OCR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # optional thresholding (comment/uncomment as needed)
    # _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(gray, lang='eng')
    text_upper = text.upper()

    # Keyword flags
    has_subtotal_word = 1 if ("SUBTOTAL" in text_upper or "SUB TOTAL" in text_upper) else 0
    has_tax_word = 1 if "TAX" in text_upper else 0
    has_total_word = 1 if "TOTAL" in text_upper else 0
    has_thank_word = 1 if "THANK" in text_upper else 0

    # Extract monetary values
    amounts = parse_amounts(text)
    # Heuristic: last value is likely total, previous might be tax/subtotal
    total_val = amounts[-1] if len(amounts) >= 1 else 0.0
    tax_val = amounts[-2] if len(amounts) >= 2 else 0.0
    subtotal_val = amounts[-3] if len(amounts) >= 3 else 0.0

    # item count heuristics: count lines with an amount + text item
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    item_line_count = 0
    for ln in lines:
        if re.search(r"\d+\.\d{2}", ln):
            # skip lines like "TOTAL:" if they contain the keyword
            if any(k in ln.upper() for k in ["TOTAL", "SUBTOTAL", "TAX", "CHANGE", "BALANCE"]):
                continue
            item_line_count += 1

    # small derived features
    numeric_length = len(amounts)
    numeric_sum = sum(amounts) if amounts else 0.0
    # check numeric consistency: does subtotal + tax ≈ total (within small epsilon)
    try:
        consistent_total = 1 if abs((subtotal_val + tax_val) - total_val) < max(1.0, 0.05 * total_val) else 0
    except:
        consistent_total = 0

    return {
        "filename": os.path.basename(image_path),
        "items_count": item_line_count,
        "numbers_found": numeric_length,
        "sum_numbers": numeric_sum,
        "subtotal_val": float(subtotal_val),
        "tax_val": float(tax_val),
        "total_val": float(total_val),
        "has_subtotal_word": has_subtotal_word,
        "has_tax_word": has_tax_word,
        "has_total_word": has_total_word,
        "has_thank_word": has_thank_word,
        "consistent_total": consistent_total,
        "raw_text": text,
        "label": label
    }

if __name__ == "__main__":
    # build dataset CSV from data/real and data/fake folders
    out_rows = []
    for folder, lab in [("data/real", 1), ("data/fake", -1)]:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, fname)
                feat = extract_bill_features(path, label=lab)
                if feat:
                    print(f"Processed: {path}")
                    out_rows.append(feat)

    if out_rows:
        df = pd.DataFrame(out_rows)
        df.to_csv("bill_features.csv", index=False)
        print("Saved bill_features.csv")
    else:
        print("No receipts found to process.")
 