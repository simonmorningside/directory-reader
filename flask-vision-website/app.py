from flask import Flask, render_template, request
import fitz

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        extracted_text += f"Page {page_num + 1} Text:\n{text}\n{'-'*40}\n"
    return extracted_text

@app.route("/", methods=["GET", "POST"])
def upload_pdf():
    if request.method == "POST":
        file = request.files.get("pdf_file")
        if file:
            file_path = "uploaded_pdf.pdf"
            file.save(file_path)
            
            extracted_text = extract_text_from_pdf(file_path)
            
            return render_template("index.html", extracted_text=extracted_text)
    
    return render_template("index.html", extracted_text=None)

if __name__ == "__main__":
    app.run(debug=True)
