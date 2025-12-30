import cv2
import pytesseract

# 👇 Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

if __name__ == "__main__":
    input_path = r"data\real\1000-receipt.jpg"
    print(extract_text(input_path))
