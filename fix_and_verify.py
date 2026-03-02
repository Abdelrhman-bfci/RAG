import os
import glob
import PyPDF2
from app.qa.rag_chain import answer_question
from app.ingestion.offline_web_ingest import ingest_offline_downloads

def find_catalog():
    downloads_dir = "offine_downloads"
    pdfs = glob.glob(os.path.join(downloads_dir, "*.pdf"))
    target_terms = ["Computer and Systems", "Computer Engineering", "Catalog", "Faculty", "Undergraduate"]
    
    print(f"Scanning {len(pdfs)} PDFs to find CS_Catalog...")
    for pdf_path in pdfs:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = "".join([reader.pages[i].extract_text() or "" for i in range(min(5, len(reader.pages)))])
                content = content.lower()
                matches = sum(1 for term in target_terms if term.lower() in content)
                
                if matches >= 2:
                    new_path = os.path.join(downloads_dir, "CS_Catalog.pdf")
                    if pdf_path != new_path and not os.path.exists(new_path):
                        print(f"Found match! Renaming {os.path.basename(pdf_path)} to CS_Catalog.pdf")
                        os.rename(pdf_path, new_path)
                        return True
                    elif pdf_path == new_path:
                        print("CS_Catalog.pdf is already correctly named.")
                        return True
        except Exception:
            pass
    print("Could not find matching CS_Catalog in offline downloads.")
    return False

def verify():
    query = "List only professors in Computer and Systems Engineering department?"
    print(f"\nVerifying Query: {query}")
    print("-" * 50)
    
    result = answer_question(query, deep_thinking=False)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Performance: {result.get('performance', 'N/A')}")
    print("\nAnswer:\n", result.get('answer', ''))
    print(f"\nSources Found: {len(result.get('sources', []))}")
    for src in result.get('sources', []):
        print(f" - {src}")

if __name__ == "__main__":
    if find_catalog():
        print("\nRe-indexing offline downloads...")
        # Since this uses standard generators
        for msg in ingest_offline_downloads():
            if msg:
                print(msg.strip())
        print("Indexing completed.")
    else:
        print("\nSkipping indexing since catalog was not found.")
    
    print("\nRunning Verification...")
    verify()
