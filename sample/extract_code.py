import json
import os

notebook_path = '/media/kisna/docker_d/unicolor/sample/sample.ipynb'
output_path = '/media/kisna/docker_d/unicolor/sample/reproduce_sample.py'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        code_cells.append(source)

full_code = '\n\n'.join(code_cells)

with open(output_path, 'w') as f:
    f.write(full_code)

print(f"Extracted {len(code_cells)} code cells to {output_path}")
