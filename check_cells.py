import json

def list_cells():
    path = "program/preprocessing_nlp.ipynb"
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Failed to open notebook: {e}")
        return
        
    for i, cell in enumerate(nb.get("cells", [])):
        source_str = "".join(cell.get("source", []))
        label = "Markdown" if cell.get("cell_type") == "markdown" else "Code"
        print(f"Index {i} [{label}]: {source_str[:50].replace(chr(10), ' ')}")

if __name__ == '__main__':
    list_cells()
