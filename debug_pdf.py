import fitz
doc = fitz.open("RESOURCE/SUTENG Bylaw Specs_merged.pdf")
found = False
for page in doc:
    text = page.get_text()
    if "Internet" in text or "Things" in text or "IoT" in text:
        print(f"--- Page {page.number} ---")
        print(text)
        found = True
        break
if not found:
    print("No mentions found.")
