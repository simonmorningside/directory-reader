import pdfplumber

pdf_path = "/Users/calepittenger/Desktop/243.pdf"  # Adjust path if necessary
output_text_path = "/Users/calepittenger/Desktop/output_text.txt"

text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n" if page.extract_text() else ""

with open(output_text_path, "w", encoding="utf-8") as text_file:
    text_file.write(text)

print("Text extraction complete. Output saved to:", output_text_path)
pip