import fitz
doc = fitz.open("RESOURCE/SUTENG Bylaw Specs_merged.pdf")
results = []
for page in doc:
    text = page.get_text()
    if "Internet of Things" in text:
        # Get more context
        results.append(f"--- Page {page.number} ---\n{text}")

if results:
    for r in results:
        print(r)
else:
    print("No full phrase mentions found.")
