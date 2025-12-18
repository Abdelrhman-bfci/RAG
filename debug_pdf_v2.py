import fitz
doc = fitz.open("RESOURCE/SUTENG Bylaw Specs_merged.pdf")
results = []
for page in doc:
    text = page.get_text()
    if "Internet of Things" in text:
        results.append(f"Page {page.number} (Full Phrase): {text[:200]}...")
    elif "IoT" in text:
        results.append(f"Page {page.number} (IoT): {text[:200]}...")

if results:
    for r in results[:10]:
        print(r)
else:
    print("No mentions found.")
