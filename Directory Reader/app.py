import os
import json
import re
import numpy as np
import cv2
import pytesseract
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from pdf2image import convert_from_path

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to Tesseract on Windows (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Convert PDF to image(s)
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=200)

# Extract text using Tesseract with clean config
def extract_text_from_image(image):
    custom_config = r'--psm 6'
    return pytesseract.image_to_string(image, config=custom_config)

# Simplified preprocessing (grayscale only)
def preprocess_crop_for_ocr(cropped_np, page):
    gray = cv2.cvtColor(cropped_np, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"uploads/cleaned_page_{page}.png", gray)
    return gray

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            images = convert_pdf_to_images(file_path)
            for i, image in enumerate(images):
                image.save(os.path.join(UPLOAD_FOLDER, f"page_{i}.png"))

            return render_template('select_region.html', image_url=f"/uploads/page_0.png", page=0, total_pages=len(images))

    return render_template('index2.html', text=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/select_region')
def show_page():
    page = int(request.args.get('page', 0))
    total_pages = len([f for f in os.listdir(UPLOAD_FOLDER) if f.startswith("page_")])

    return render_template('select_region.html',
                           image_url=f"/uploads/page_{page}.png",
                           page=page,
                           total_pages=total_pages)

@app.route('/extract_region', methods=['POST'])
def extract_region():
    try:
        def auto_quote_addresses_in_blocks(text):
            blocks = [block.strip() for block in text.strip().split('\n\n') if block.strip()]
            quoted_blocks = []
            for block in blocks:
                if block.isupper():
                    quoted_blocks.append(block)
                    continue
                match = re.search(r'\b(?:[rh]\d+|\d{2,5})', block)
                if match:
                    idx = match.start()
                    name = block[:idx].strip()
                    address = block[idx:].strip()
                    quoted_blocks.append(f'{name} [[ADDR: {address}]]')
                else:
                    quoted_blocks.append(block)
            return '\n\n'.join(quoted_blocks)

        # Get polygon region
        polygon_data = request.form['polygon']
        page = int(request.form['page'])
        polygon = json.loads(polygon_data)

        if not polygon:
            return "No polygon points submitted."

        # Load and crop image
        image_path = os.path.join(UPLOAD_FOLDER, f"page_{page}.png")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        pts = np.array([[p['x'], p['y']] for p in polygon], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(image_np, image_np, mask=mask)

        # Split left and right
        x, y, w, h = cv2.boundingRect(pts)
        mid_x = x + w // 2
        left_crop = masked[y:y+h, x:mid_x]
        right_crop = masked[y:y+h, mid_x:x+w]

        # OCR
        left_processed = preprocess_crop_for_ocr(left_crop, page)
        right_processed = preprocess_crop_for_ocr(right_crop, page)

        left_text = extract_text_from_image(Image.fromarray(left_processed))
        right_text = extract_text_from_image(Image.fromarray(right_processed))

        # Combine and quote
        combined_text = left_text.strip() + "\n\n" + right_text.strip()
        quoted_text = auto_quote_addresses_in_blocks(combined_text)
        print("--- OCR OUTPUT ---")
        print(combined_text[:500])
        print("--- QUOTED OUTPUT ---")
        print(quoted_text[:500])


        # ✅ This MUST be named `text` for the textarea in review_text.html
        return render_template('review_text.html', text=quoted_text)

    except Exception as e:
        return f"Error processing polygon: {e}"
    

@app.route('/finalize_text', methods=['POST'])
def finalize_text():
    raw_text = request.form['final_text']

    # Split entries by the END of the address block
    chunks = raw_text.split(']]')
    entries = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Add ]] back since split removed it
        chunk += ']]'

        if '[[ADDR:' in chunk:
            try:
                name_part, addr_part = chunk.split('[[ADDR:', 1)
                name = name_part.strip().replace('\n', ' ')
                address = addr_part.strip().rstrip(']').strip(':').strip()
                entries.append({'name': name, 'address': address})
            except Exception:
                entries.append({'name': chunk.strip(), 'address': ''})
        else:
            entries.append({'name': chunk.strip(), 'address': ''})

    return render_template('finalize_edit.html', entries=entries)

@app.route('/save_csv', methods=['POST'])
def save_csv():
    from flask import Response
    import csv
    from io import StringIO

    names = request.form.getlist('name')
    addresses = request.form.getlist('address')

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['Name', 'Address'])
    for name, address in zip(names, addresses):
        writer.writerow([name.strip(), address.strip()])

    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=extracted_data.csv'}
    )

if __name__ == '__main__':
    app.run(debug=True)