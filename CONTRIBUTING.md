# Contributing to PDUUT

Thanks for your interest in contributing! PDUUT (PDF Data Unification and Understanding Tool) welcomes issues, discussions, and pull requests.

## Ways to contribute
- **Bug reports**: Use the Bug Report template and include logs, sample PDFs, steps to reproduce.
- **Feature requests**: Use the Feature Request template, describe the use‑case and expected UX.
- **Docs**: Improve README, examples, or troubleshooting.
- **Code**: Fix bugs, improve extraction accuracy, performance, tests, or add new export formats.

## Development setup
1. Clone the repo and create a virtual env:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. System deps:
   - Tesseract OCR (required)
   - libGL on Linux for OpenCV
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Code style and practices
- Keep modules small and testable. Add logging where exceptions are caught.
- Prefer pure functions for utilities. Avoid global state.
- Handle platform differences (Linux/macOS/Windows) gracefully.
- Be defensive with imports of PyMuPDF (use `pymupdf as fitz` and fallback to `fitz`).
- Avoid breaking changes; discuss in an issue first.

## Pull request process
1. Fork and create a feature branch from `main`.
2. Ensure README or docs are updated if behavior changes.
3. Add/adjust tests if applicable (coming soon).
4. Run a basic smoke test with a few PDFs.
5. Open a PR using the template. Link to related issues and provide before/after notes or screenshots.

## Issue triage
- Reproducible bugs with clear steps are prioritized.
- Performance and accuracy improvements are welcome.
- Feature requests are discussed and labeled for scope/complexity.

## Community expectations
We follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and constructive. We value diverse perspectives and clear, actionable feedback.

## Need help?
Open a discussion or issue with details. Maintainers will guide you on where to start.

## Maintainers
- Mohsin Raja (project author/maintainer)
