import sys
from pathlib import Path
from pdfminer.high_level import extract_text


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_pdf_text.py <input_pdf_path> [output_txt_path]", file=sys.stderr)
        sys.exit(1)

    input_pdf = Path(sys.argv[1])
    if not input_pdf.exists():
        print(f"Input PDF not found: {input_pdf}", file=sys.stderr)
        sys.exit(2)

    output_txt = None
    if len(sys.argv) >= 3:
        output_txt = Path(sys.argv[2])
    else:
        # Default output path under docs with same base name
        output_dir = Path("docs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_txt = output_dir / (input_pdf.stem + ".txt")

    try:
        text = extract_text(str(input_pdf))
    except Exception as e:
        print(f"Failed to extract text: {e}", file=sys.stderr)
        sys.exit(3)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(text, encoding="utf-8")
    print(f"Wrote extracted text to: {output_txt}")


if __name__ == "__main__":
    main()
