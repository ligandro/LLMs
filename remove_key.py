import json
import sys

# Read the notebook
with open('Topics/Vector_Databases/db_1.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and replace the API key in cells
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Convert to list if it's a string
        if isinstance(source, str):
            source = [source]
        
        # Replace the line with the API key
        new_source = []
        for line in source:
            if 'KEY = "sk-proj-' in line:
                new_source.append('KEY = "your-api-key-here"  # Replace with your OpenAI API key\n')
            else:
                new_source.append(line)
        
        cell['source'] = new_source

# Write back
with open('Topics/Vector_Databases/db_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ API key removed from notebook")
