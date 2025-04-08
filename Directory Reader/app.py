import os
from flask import Flask, request, render_template, send_from_directory
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# If you're on Windows and Tesseract is not in your PATH, set the path manually:
# For example, on Windows, if Tesseract is installed in 'C:\Program Files\Tesseract-OCR\':
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_np):
    import cv2
    import numpy as np

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 2. Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Remove small dots with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Optional: remove small specks (contours under a certain size)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:  # Remove specks smaller than 10 pixels
            cv2.drawContours(cleaned, [cnt], 0, 0, -1)

    # 5. Invert image back for OCR (black text on white background)
    final = cv2.bitwise_not(cleaned)

    return final

# Step 1: Convert PDF to images
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=200)  # Higher DPI for better resolution

# Step 2: Apply OCR to an image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Step 3: Handle file upload
        file = request.files['file']
        if file:
            # Clear uploads dir
            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Step 4: Convert PDF to images
            images = convert_pdf_to_images(file_path)

            # Save all images for navigation
            for i, image in enumerate(images):
                image_path = os.path.join(UPLOAD_FOLDER, f"page_{i}.png")
                image.save(image_path)

            # Start on page 0
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

import json
import numpy as np
from flask import redirect, url_for
import cv2  # OpenCV
from PIL import Image
import os

@app.route('/extract_region', methods=['POST'])
def extract_region():
    try:
        import re

        def preprocess_crop_for_ocr(cropped_np, page):
            """Preprocess the image without deskewing, just threshold and clean."""
            # Convert to grayscale
            gray = cv2.cvtColor(cropped_np, cv2.COLOR_BGR2GRAY)

            # Save grayscale image for inspection
            cv2.imwrite(f"uploads/grayscale_page_{page}.png", gray)

            # Threshold (for better OCR)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Save thresholded image for inspection
            cv2.imwrite(f"uploads/threshold_page_{page}.png", thresh)

            # Clean up small noise (optional: morphological cleaning)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # Save cleaned image for inspection
            cv2.imwrite(f"uploads/cleaned_page_{page}.png", cleaned)

            # Return the cleaned image (no deskewing)
            return cleaned

        # Get polygon + page number
        polygon_data = request.form['polygon']
        page = int(request.form['page'])
        polygon = json.loads(polygon_data)

        if not polygon:
            return "No polygon points submitted."

        # Load the original image
        image_path = os.path.join(UPLOAD_FOLDER, f"page_{page}.png")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Create and apply mask
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        pts = np.array([[p['x'], p['y']] for p in polygon], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(image_np, image_np, mask=mask)

        # Get bounding box of polygon and crop
        x, y, w, h = cv2.boundingRect(pts)
        cropped = masked[y:y+h, x:x+w]

        # Preprocess cropped image (no deskewing)
        preprocessed = preprocess_crop_for_ocr(cropped, page)
        result_image = Image.fromarray(preprocessed)

        # OCR
        raw_text = extract_text_from_image(result_image)

        # Parse name-address pairs
        entries = []
        for line in raw_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.search(r'\b\d', line)
            if match:
                idx = match.start()
                name = line[:idx].strip()
                address = line[idx:].strip()
                entries.append({'name': name, 'address': address})

        # HTML table
        table_html = "<table border='1' cellpadding='5'><tr><th>Name</th><th>Address</th></tr>"
        for entry in entries:
            table_html += f"<tr><td>{entry['name']}</td><td>{entry['address']}</td></tr>"
        table_html += "</table>"

        return render_template('review_text.html', text=table_html)

    except Exception as e:
        return f"Error processing polygon: {e}"


@app.route('/finalize_text', methods=['POST'])
def finalize_text():
    from bs4 import BeautifulSoup

    raw_html = request.form['final_text']
    soup = BeautifulSoup(raw_html, 'html.parser')

    entries = []
    for row in soup.find_all('tr')[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) != 2:
            continue

        name = cols[0].get_text(strip=True)
        address = cols[1].get_text(strip=True)

        if name or address:
            entries.append({'name': name, 'address': address})

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
        writer.writerow([name, address])

    csv_output = si.getvalue()
    return Response(
        csv_output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=extracted_data.csv'}
    )



if __name__ == '__main__':
    # Ensure 'uploads' folder exists for saving PDFs
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
