# PDUUT - PDF Data Extraction Tool

## Overview

PDUUT (PDF Data Extraction Tool) is a specialized Streamlit application designed to extract structured, page-wise data from educational book PDFs for Retrieval-Augmented Generation (RAG) systems. The tool processes PDF documents to extract text, images, tables, and mathematical equations, organizing the output in a structured format suitable for machine learning and AI applications.

The application uses multiple processing engines to handle different types of content:
- OCR engine for text extraction from images
- Image extractor for diagrams and figures
- Table detector for structured data
- Equation detector for mathematical content

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with a clean, intuitive interface
- **Layout**: Wide layout with expandable sidebar for configuration options
- **User Controls**: File upload, language selection, processing options toggles
- **Progress Tracking**: Real-time progress callbacks during PDF processing

### Backend Architecture
- **Core Processor**: `PDFProcessor` class orchestrates the entire extraction pipeline
- **Modular Design**: Separate engines for different content types (OCR, images, tables, equations)
- **PDF Library**: PyMuPDF (fitz) for PDF document manipulation and content extraction
- **Computer Vision**: OpenCV and PIL for image processing and analysis

### Content Extraction Pipeline
1. **PDF Parsing**: Extract metadata and page-level content using PyMuPDF
2. **Text Extraction**: OCR processing with Tesseract for image-based text
3. **Image Processing**: Extract embedded images and detect diagram regions
4. **Table Detection**: Computer vision-based table region identification and data extraction
5. **Equation Recognition**: Pattern matching and image analysis for mathematical content

### Processing Engines
- **OCREngine**: Multi-language text extraction with preprocessing and cleaning
- **ImageExtractor**: Embedded image extraction and computer vision-based diagram detection
- **TableDetector**: Table region identification and structured data extraction
- **EquationDetector**: Mathematical equation detection using pattern matching and image analysis

### Output Structure
- **Hierarchical Organization**: Page-wise data with categorized content types
- **Asset Management**: Separate directories for images, tables, and text files
- **Metadata Preservation**: PDF metadata and extraction statistics included
- **JSON Format**: Structured output compatible with RAG systems

### File Management
- **Temporary Processing**: Safe temporary file handling during extraction
- **Asset Organization**: Organized directory structure with sanitized filenames
- **ZIP Output**: Compressed delivery of processed results

## External Dependencies

### Core Libraries
- **PyMuPDF (fitz)**: PDF document processing and content extraction
- **Streamlit**: Web application framework for user interface
- **OpenCV**: Computer vision operations for image and table detection
- **Pillow (PIL)**: Image processing and format conversion
- **NumPy**: Numerical operations and array processing

### OCR and Text Processing
- **Pytesseract**: OCR engine integration for text extraction from images
- **Tesseract OCR**: Underlying OCR engine (system dependency)

### Data Processing
- **Pandas**: Table data manipulation and CSV export
- **JSON**: Structured data serialization and storage

### System Integration
- **OS Libraries**: File system operations and directory management
- **Tempfile**: Temporary file and directory handling
- **Zipfile**: Archive creation for output packaging
- **Datetime**: Timestamp generation and metadata

### Optional Enhancements
- **Regular Expressions**: Pattern matching for equation detection
- **Multiple Language Support**: OCR processing in various languages (English, Spanish, French, German, Italian, Portuguese)