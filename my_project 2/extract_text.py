import pdfplumber
import pandas as pd
import re

def extract_data_from_pdf(pdf_path):
    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    # Exclude advertisements by filtering lines with excessive capitalization or non-directory formats
                    if not re.search(r"[A-Z\s]{5,}", line) and not any(x in line for x in ["PHONE", "Tel", "Sales", "Co.", "Inc.", "Company"]):
                        match = re.match(r"([A-Za-z]+(?: [A-Za-z]+)*),? (.*?) (h\d+ .*|r\d+ .*|r.*)", line)
                        if match:
                            name = match.group(1).strip()
                            address = match.group(2).strip()
                            business = match.group(3).strip() if match.group(3) else "-"
                            extracted_data.append([name, address, business])

    return extracted_data

def print_to_terminal(data):
    df = pd.DataFrame(data, columns=["Name", "Address", "Business"])
    print(df.to_string(index=False))

def save_to_csv(data, output_file):
    df = pd.DataFrame(data, columns=["Name", "Address", "Business"])
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    pdf_path = "/home/cwp002/my_project/243.pdf"
    output_file = "/home/cwp002/my_project/extracted_data.csv"
    data = extract_data_from_pdf(pdf_path)
    print_to_terminal(data)
    save_to_csv(data, output_file)
