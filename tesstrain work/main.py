import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ========== CONFIG ==========
pdf_path = r"C:\Users\Morningside\Downloads\trainingpages.pdf"
output_dir = r"C:\Users\Morningside\directory-reader\trainingdata"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tess_lang = "eng"  # or use 'eng+custom' if fine-tuning
dpi = 300  # high-quality images
max_pages = 50
# =============================

os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)

for i, image in enumerate(images, start=1):
    base_filename = f"page_{i:02d}"
    image_path = os.path.join(output_dir, base_filename + ".png")
    txt_path = os.path.join(output_dir, base_filename + ".gt.txt")

    # Save image
    image.save(image_path, "PNG")

    # Generate text using Tesseract OCR (as GT baseline)
    text = pytesseract.image_to_string(image, lang=tess_lang)

    # Save ground truth
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"Saved: {image_path}, {txt_path}")
