import os
import re
import json

directory = '/home/asuengai/RAG/RAG/offine_downloads'
search_term = 'Computer and Systems Engineering'
output_file = '/home/asuengai/RAG/RAG/professors.json'

professors = []

if not os.path.exists(directory):
    print(f"Directory {directory} not found.")
    exit(1)

for filename in os.listdir(directory):
    if filename.endswith('.html'):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if search_term in content:
                    # Clean content of some HTML tags to make regex more robust
                    content_clean = re.sub(r'\s+', ' ', content)
                    
                    # Try to find name in various tags
                    name = None
                    # instructorName class
                    name_match = re.search(r'<h3 class="instructorName"[^>]*>(.*?)</h3>', content_clean)
                    if name_match:
                        name = name_match.group(1).strip()
                    
                    # panel-title class
                    if not name:
                        name_match = re.search(r'<h3 class="panel-title"[^>]*>(.*?)</h3>', content_clean)
                        if name_match:
                            name = name_match.group(1).strip()
                    
                    # h4 instructorName2
                    title = "Unknown Title"
                    title_match = re.search(r'<h4 class="instructorName2"[^>]*>(.*?)</h4>', content_clean)
                    if title_match:
                        title = title_match.group(1).strip()
                    
                    if name:
                        # Clean up name if it still has tags
                        name = re.sub(r'<[^>]+>', '', name).strip()
                        title = re.sub(r'<[^>]+>', '', title).strip()
                        professors.append({
                            "name": name, 
                            "title": title, 
                            "department": search_term,
                            "source_file": filename
                        })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Deduplicate by name
unique_profs = {}
for prof in professors:
    unique_profs[prof['name']] = prof

professors_list = sorted(unique_profs.values(), key=lambda x: x['name'])

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(professors_list, f, indent=4, ensure_ascii=False)

print(f"Successfully extracted {len(professors_list)} professors to {output_file}")
for prof in professors_list:
    print(f"- {prof['name']} ({prof['title']})")
