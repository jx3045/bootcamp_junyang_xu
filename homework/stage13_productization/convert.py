import nbformat
import os
import glob

# Directory containing ipynb files (default: current folder)
input_dir = "."

# Loop over all ipynb files in the folder
for ipynb_file in glob.glob(os.path.join(input_dir, "*.ipynb")):
    py_file = os.path.splitext(ipynb_file)[0] + ".py"
    print(f"Converting: {ipynb_file} -> {py_file}")

    # Load notebook
    with open(ipynb_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Extract code cells only
    code_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source.strip()
            if source:  # skip empty code cells
                code_cells.append(source)

    # Write to .py file
    with open(py_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from Jupyter Notebook\n")
        f.write("# Only code cells preserved (markdown/outputs removed)\n\n")
        f.write("\n\n".join(code_cells))  # separate cells by blank line

print("✅ All notebooks converted successfully.")
