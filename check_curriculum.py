import fitz
doc = fitz.open("RESOURCE/SUTENG Bylaw Specs_merged.pdf")
results = []
for i in range(90, 120): # Focus on curriculum pages
    if i < len(doc):
        text = doc[i].get_text()
        if "Things" in text or "IoT" in text:
            results.append(f"--- Page {i} ---\n{text}")

if results:
    for r in results:
        print(r)
else:
    print("No mentions found in range 90-120.")
