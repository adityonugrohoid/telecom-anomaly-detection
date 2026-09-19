"""Export chosen image outputs from an executed notebook as PNG files.

This script reads the notebook JSON directly and writes the requested cell
outputs to disk. It does not execute the notebook, so the notebook must
already have been run and saved with its outputs intact. It fails with a
clear error if a requested cell does not exist, is not a code cell, has no
image output, or has more than one image output (ambiguous selection).
"""

import argparse
import base64
import json
from pathlib import Path


def load_notebook(notebook_path: Path) -> dict:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    with open(notebook_path, encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def extract_cell_image(notebook: dict, cell_index: int) -> bytes:
    cells = notebook.get("cells", [])
    if cell_index < 0 or cell_index >= len(cells):
        raise IndexError(
            f"Cell index {cell_index} is out of range for a notebook with {len(cells)} cells"
        )

    cell = cells[cell_index]
    if cell.get("cell_type") != "code":
        raise ValueError(f"Cell {cell_index} is a '{cell.get('cell_type')}' cell, not code")

    png_outputs = [
        output["data"]["image/png"]
        for output in cell.get("outputs", [])
        if "data" in output and "image/png" in output["data"]
    ]

    if not png_outputs:
        raise ValueError(
            f"Cell {cell_index} has no image/png output. "
            "The notebook must be executed and saved with outputs before exporting figures."
        )
    if len(png_outputs) > 1:
        raise ValueError(
            f"Cell {cell_index} has {len(png_outputs)} image/png outputs. "
            "This script exports exactly one image per cell; choose a cell with a single figure."
        )

    b64_data = png_outputs[0]
    if isinstance(b64_data, list):
        b64_data = "".join(b64_data)
    return base64.b64decode(b64_data)


def parse_figure_spec(spec: str) -> tuple[int, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid --figure value '{spec}'. Expected format 'cell_index:filename'.")
    cell_index_str, filename = spec.split(":", 1)
    if not filename:
        raise ValueError(f"Invalid --figure value '{spec}'. Filename is empty.")
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    return int(cell_index_str), filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export chosen image outputs from an executed notebook as PNG files."
    )
    parser.add_argument("notebook_path", type=Path, help="Path to the executed .ipynb file")
    parser.add_argument("output_dir", type=Path, help="Directory to write PNG files into")
    parser.add_argument(
        "--figure",
        action="append",
        required=True,
        dest="figures",
        metavar="CELL_INDEX:FILENAME",
        help=(
            "Cell index and output filename to export, for example "
            "12:anomaly-detection-timeline. Repeat this flag to export more than one figure."
        ),
    )
    args = parser.parse_args()

    notebook = load_notebook(args.notebook_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.figures:
        cell_index, filename = parse_figure_spec(spec)
        image_bytes = extract_cell_image(notebook, cell_index)
        output_path = args.output_dir / filename
        output_path.write_bytes(image_bytes)
        print(f"Wrote cell {cell_index} image to {output_path} ({len(image_bytes):,} bytes)")


if __name__ == "__main__":
    main()
