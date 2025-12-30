import cv2
import os

def preprocess_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(save_path, thresh)
    print(f"✅ Saved processed: {save_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\Admin\OneDrive\Desktop\AutoAudit\data\real"
    output_folder = r"C:\Users\Admin\OneDrive\Desktop\AutoAudit\data\processed"

    # make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            preprocess_image(input_path, output_path)
