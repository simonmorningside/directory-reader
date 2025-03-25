from flask import Flask, render_template, request
import pytesseract
from pdf2image import convert_from_path
import os

app = Flask(__name__)

# Create an upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define paths for Poppler and Tesseract
POPPLER_PATH = "/usr/bin/"  # Update if necessary
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Update if necessary

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = None

    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "No file part"

        file = request.files["pdf_file"]
        if file.filename == "":
            return "No selected file"

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            try:
                # Convert PDF to images with explicit poppler path
                images = convert_from_path(file_path, poppler_path=POPPLER_PATH)

                # Apply OCR and extract text
                extracted_text = "\n".join([pytesseract.image_to_string(image) for image in images])

            except Exception as e:
                extracted_text = f"Error: {str(e)}"

    return render_template("index.html", extracted_text=extracted_text)

if __name__ == "__main__":
    app.run(debug=True)
