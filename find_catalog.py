import os
import glob
import PyPDF2

def search_pdfs():
    downloads_dir = "offine_downloads"
    pdfs = glob.glob(os.path.join(downloads_dir, "*.pdf"))
    print(f"Searching {len(pdfs)} PDF files...")
    
    target_terms = ["Computer and Systems", "Computer Engineering", "Catalog", "Faculty", "Undergraduate"]
    
    for pdf_path in pdfs:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                # Check first 5 pages for title info
                pages_to_check = min(5, num_pages)
                
                content = ""
                for i in range(pages_to_check):
                    content += reader.pages[i].extract_text() or ""
                
                content = content.lower()
                matches = sum(1 for term in target_terms if term.lower() in content)
                
                if matches >= 2:
                    print(f"\nPotential Match found: {pdf_path}")
                    print(f"Score: {matches} terms matched")
                    print(f"Content preview: {content[:100]}...")
                    # Let's rename it to make it obvious
                    new_path = os.path.join(downloads_dir, "CS_Catalog.pdf")
                    if not os.path.exists(new_path):
                        print(f"Renaming {pdf_path} to {new_path}")
                        os.rename(pdf_path, new_path)
                        break
        except Exception as e:
            pass

if __name__ == "__main__":
    search_pdfs()
