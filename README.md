# PDUUT — PDF Data Unification and Understanding Tool

Extract structured, page‑wise data from educational PDFs for downstream RAG systems. PDUUT (PDF Data Unification and Understanding Tool) unifies text, images/diagrams, tables, and equations into consistent outputs. Built with Streamlit, PyMuPDF, Tesseract OCR, and OpenCV.

## Features
- **Page‑wise extraction**: text, images/diagrams, tables, and math equations per page (`PDFProcessor`).
- **OCR fallback + fusion**: native PDF text + Tesseract OCR via `OCREngine` with optional advanced preprocessing.
- **Batch processing**: parallel multi‑PDF pipeline with progress and consolidated stats (`BatchPDFProcessor`).
- **Quality assessment**: basic, fast heuristics for RAG suitability (`QualityAssessment`).
- **Multi‑format export**: JSON (flat + page‑wise), CSV, XML, Markdown, YAML, HTML (`ExportManager`).
- **Self‑contained results package**: assets and metadata zipped for download from the UI.

## Quickstart

### 1) Prerequisites
- Python 3.11+
- System packages
  - Tesseract OCR (required by `pytesseract`)
  - libGL (for OpenCV on Linux)

Linux (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libgl1
```

macOS (Homebrew):
```bash
brew install tesseract
```

Windows:
- Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
- Ensure the Tesseract binary is on PATH (or set `pytesseract.pytesseract.tesseract_cmd`).

### 2) Install Python deps
From the project root `Pduut/`:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install pymupdf opencv-python pytesseract pillow numpy pandas streamlit pyyaml markdown
```
Note: `pyproject.toml` lists the project deps if you prefer a tool like `uv`/`pip-tools`.

### 3) Run the app
```bash
streamlit run app.py
```
The app opens at http://localhost:8501

## How it works
- `app.py`: Streamlit UI. Choose Single PDF or Batch, toggle extraction options, pick export formats, and run.
- `pdf_processor.py`: Orchestrates page rendering with PyMuPDF and runs text/OCR, image, table, and equation extraction.
- `ocr_engine.py`: Handles OCR (Tesseract) with light OpenCV preprocessing and optional advanced pipeline.
- `batch_processor.py`: Threads multiple documents and aggregates stats; can export consolidated results.
- `export_manager.py`: Writes results to JSON/CSV/XML/Markdown/YAML/HTML and can bundle a ZIP.
- `utils.py`: Helpers for output structure, filenames, summaries, and validation.

## Using the app
- **Single PDF**
  1. Upload a `.pdf`.
  2. Select OCR languages (e.g., `eng`, `spa`, ...).
  3. Choose what to extract: Images, Tables, Equations.
  4. Optional: Advanced OCR preprocessing; Quality Assessment.
  5. Pick export formats (default: JSON).
  6. Click “Process PDF” and download the ZIP.

- **Batch Processing**
  1. Upload multiple `.pdf` files.
  2. Set parallel workers.
  3. Options mirror single‑file processing.
  4. Start and download the batch ZIP when complete.

## Outputs
- A downloadable ZIP containing:
  - `extraction_result.json` (and `flat_structure.json`, plus `pages_json/`)
  - Optional exports: `csv_export/`, `extraction_result.xml`, `extraction_result.md`, `extraction_result.yaml`, `extraction_result.html`
  - `assets/`
    - `images/` — extracted diagrams
    - `tables/` — detected table JSON/CSV where available
    - `text/` — `page_{n}.txt`
- In‑app metrics summarize pages, text length, images, tables, and equations. A JSON preview is provided.

## Configuration (sidebar)
- **Processing Mode**: Single PDF | Batch
- **OCR Languages**: e.g., `eng`, `spa`, `fra`, `deu`, `ita`, `por`
- **Options**: Extract Images | Extract Tables | Detect Equations
- **Advanced**: Advanced OCR Preprocessing | Quality Assessment
- **Export Formats**: JSON, CSV, XML, Markdown, YAML, HTML

## Quality Assessment
`QualityAssessment.assess_extraction_quality()` computes quick heuristics:
- Text, Image, Table, Equation quality scores
- Overall weighted score and recommendations
Use this to gauge RAG‑readiness and decide whether rescans or parameter tweaks are needed.

## Project structure
```
Pduut/
├─ app.py                   # Streamlit UI
├─ pdf_processor.py         # Core per‑page pipeline
├─ ocr_engine.py            # Tesseract + preprocessing
├─ image_extractor.py       # Diagram/image extraction (used by PDFProcessor)
├─ table_detector.py        # Table detection + export (used by PDFProcessor)
├─ equation_detector.py     # Equation detection + LaTeX (used by PDFProcessor)
├─ batch_processor.py       # Multi‑PDF processing + summary/export
├─ export_manager.py        # Export to JSON/CSV/XML/MD/YAML/HTML
├─ utils.py                 # Helpers: structure, validation, summaries
├─ pyproject.toml           # Python dependencies
└─ .streamlit/              # Streamlit config (optional)
```

## Troubleshooting
- **Tesseract not found**: Install it and ensure it’s on PATH. Alternatively set `pytesseract.pytesseract.tesseract_cmd` in `ocr_engine.py`.
- **OpenCV libGL error on Linux**: `sudo apt-get install -y libgl1`.
- **Slow OCR / low accuracy**: Enable “Advanced OCR Preprocessing”, provide higher‑resolution PDFs, add correct OCR languages.
- **Large PDFs**: Increase `Parallel Workers` in batch mode; consider more RAM/CPU.

## Development
- Code style: keep modules small and testable. Add logging where exceptions are caught.
- Extending exports: add a new handler in `ExportManager` and wire it in `export_data()`.
- Adding detectors: implement a detector and plug it into `PDFProcessor.process_page()`.

## License
MIT (or your preferred license)

## Acknowledgements
- PyMuPDF, Tesseract OCR, OpenCV, Streamlit, NumPy, Pandas, PyYAML, Python Markdown.
