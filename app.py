import streamlit as st
import os
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from pdf_processor import PDFProcessor
from batch_processor import BatchPDFProcessor, QualityAssessment
from export_manager import ExportManager
from utils import create_output_structure, sanitize_filename
import traceback

# Page configuration
st.set_page_config(
    page_title="PDUUT - PDF Data Extraction Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📚 PDUUT - PDF Data Extraction Tool")
    st.markdown("**Extract structured, page-wise data from educational book PDFs for RAG systems**")
    
    # Sidebar for configuration
    st.sidebar.header("Processing Mode")
    processing_mode = st.sidebar.radio(
        "Select Processing Mode",
        options=["Single PDF", "Batch Processing"],
        help="Choose between processing a single PDF or multiple PDFs at once"
    )
    
    st.sidebar.header("Configuration")
    
    # OCR Language selection
    ocr_languages = st.sidebar.multiselect(
        "OCR Languages",
        options=["eng", "spa", "fra", "deu", "ita", "por"],
        default=["eng"],
        help="Select languages for OCR processing"
    )
    
    # Processing options
    st.sidebar.header("Processing Options")
    extract_images = st.sidebar.checkbox("Extract Images/Diagrams", value=True)
    extract_tables = st.sidebar.checkbox("Extract Tables", value=True)
    extract_equations = st.sidebar.checkbox("Detect Equations", value=True)
    
    # Advanced OCR options
    st.sidebar.header("Advanced Options")
    use_advanced_ocr = st.sidebar.checkbox("Use Advanced OCR Preprocessing", value=False, 
                                          help="Apply advanced image preprocessing for better OCR results")
    quality_assessment = st.sidebar.checkbox("Enable Quality Assessment", value=True,
                                           help="Assess extraction quality for RAG applications")
    
    # Export options
    st.sidebar.header("Export Formats")
    export_formats = st.sidebar.multiselect(
        "Select Export Formats",
        options=["json", "csv", "xml", "markdown", "yaml", "html"],
        default=["json"],
        help="Choose output formats for the extracted data"
    )
    
    # File upload based on processing mode
    if processing_mode == "Single PDF":
        st.header("Upload PDF File")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload an educational PDF book for processing"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            # Process button
            if st.button("🚀 Process PDF", type="primary"):
                process_single_pdf(
                    uploaded_file, 
                    ocr_languages, 
                    extract_images, 
                    extract_tables, 
                    extract_equations,
                    use_advanced_ocr,
                    quality_assessment,
                    export_formats
                )
    
    else:  # Batch Processing
        st.header("Upload Multiple PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple educational PDF books for batch processing"
        )
        
        if uploaded_files:
            # Display files info
            total_size = sum(f.size for f in uploaded_files)
            st.info(f"**Files:** {len(uploaded_files)} | **Total Size:** {total_size / 1024 / 1024:.2f} MB")
            
            # Batch processing options
            max_workers = st.sidebar.slider("Parallel Workers", 1, 4, 2, 
                                           help="Number of PDFs to process simultaneously")
            
            # Process button
            if st.button("🚀 Process All PDFs", type="primary"):
                process_batch_pdfs(
                    uploaded_files,
                    ocr_languages, 
                    extract_images, 
                    extract_tables, 
                    extract_equations,
                    use_advanced_ocr,
                    quality_assessment,
                    export_formats,
                    max_workers
                )

def process_single_pdf(uploaded_file, ocr_languages, extract_images, extract_tables, extract_equations, 
                      use_advanced_ocr, quality_assessment, export_formats):
    """Process the uploaded PDF file"""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize processor
            processor = PDFProcessor(
                ocr_languages=ocr_languages,
                extract_images=extract_images,
                extract_tables=extract_tables,
                extract_equations=extract_equations,
                use_advanced_ocr=use_advanced_ocr
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process PDF
            status_text.text("Initializing PDF processing...")
            progress_bar.progress(10)
            
            result = processor.process_pdf(
                pdf_path, 
                temp_dir, 
                progress_callback=lambda progress, message: update_progress(progress_bar, status_text, progress, message)
            )
            
            if result:
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Quality assessment if enabled
                if quality_assessment:
                    status_text.text("Assessing extraction quality...")
                    qa = QualityAssessment()
                    quality_report = qa.assess_extraction_quality(result)
                    result['quality_assessment'] = quality_report
                
                # Export in multiple formats if requested
                if len(export_formats) > 1 or 'json' not in export_formats:
                    status_text.text("Exporting in multiple formats...")
                    export_manager = ExportManager()
                    export_results = export_manager.export_data(result, temp_dir, export_formats)
                    result['export_results'] = export_results
                
                # Display results
                display_results(result, temp_dir, uploaded_file.name)
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.error("Detailed error:")
            st.code(traceback.format_exc())

def process_batch_pdfs(uploaded_files, ocr_languages, extract_images, extract_tables, extract_equations,
                      use_advanced_ocr, quality_assessment, export_formats, max_workers):
    """Process multiple PDF files in batch"""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepare file list
            file_list = []
            for uploaded_file in uploaded_files:
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'data': uploaded_file.getvalue()
                }
                file_list.append(file_info)
            
            # Prepare processing options
            processing_options = {
                'ocr_languages': ocr_languages,
                'extract_images': extract_images,
                'extract_tables': extract_tables,
                'extract_equations': extract_equations,
                'use_advanced_ocr': use_advanced_ocr
            }
            
            # Initialize batch processor
            batch_processor = BatchPDFProcessor(max_workers=max_workers)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process batch
            status_text.text("Starting batch processing...")
            batch_results = batch_processor.process_multiple_pdfs(
                file_list, 
                processing_options,
                progress_callback=lambda progress, message: update_progress(progress_bar, status_text, progress, message)
            )
            
            if batch_results and batch_results['results']:
                progress_bar.progress(100)
                status_text.text("Batch processing complete!")
                
                # Quality assessment for batch if enabled
                if quality_assessment:
                    status_text.text("Assessing batch quality...")
                    qa = QualityAssessment()
                    for result in batch_results['results']:
                        quality_report = qa.assess_extraction_quality(result)
                        result['quality_assessment'] = quality_report
                
                # Export batch results
                if len(export_formats) > 1 or 'json' not in export_formats:
                    status_text.text("Exporting batch results...")
                    export_manager = ExportManager()
                    batch_export_path = batch_processor.create_batch_export(batch_results, temp_dir)
                    batch_results['export_package'] = batch_export_path
                
                # Display batch results
                display_batch_results(batch_results, temp_dir)
                
        except Exception as e:
            st.error(f"Error processing PDF batch: {str(e)}")
            st.error("Detailed error:")
            st.code(traceback.format_exc())

def update_progress(progress_bar, status_text, progress, message):
    """Update progress bar and status text"""
    progress_bar.progress(progress)
    status_text.text(message)

def display_results(result, temp_dir, original_filename):
    """Display processing results and provide download options"""
    
    st.success("PDF processed successfully!")
    
    # Results summary
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Pages", result.get("pdf_metadata", {}).get("pages", 0))
    
    with col2:
        total_text = sum(len(page.get("text", "")) for page in result.get("pages", []))
        st.metric("Total Text Characters", f"{total_text:,}")
    
    with col3:
        total_images = sum(len(page.get("diagrams", [])) for page in result.get("pages", []))
        st.metric("Images Extracted", total_images)
    
    with col4:
        total_tables = sum(len(page.get("tables", [])) for page in result.get("pages", []))
        st.metric("Tables Detected", total_tables)
    
    with col5:
        total_equations = sum(len(page.get("equations", [])) for page in result.get("pages", []))
        st.metric("Equations Found", total_equations)
    
    # Quality Assessment Display
    if result.get('quality_assessment'):
        st.header("📊 Quality Assessment")
        quality = result['quality_assessment']
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{quality.get('overall_score', 0):.1f}%")
        with col2:
            st.metric("Text Quality", f"{quality.get('text_quality', 0):.1f}%")
        with col3:
            st.metric("Image Quality", f"{quality.get('image_quality', 0):.1f}%") 
        with col4:
            st.metric("Table Quality", f"{quality.get('table_quality', 0):.1f}%")
        
        # Issues and recommendations
        if quality.get('issues'):
            st.warning("**Issues Found:**")
            for issue in quality['issues']:
                st.write(f"• {issue}")
        
        if quality.get('recommendations'):
            st.info("**Recommendations:**")
            for rec in quality['recommendations']:
                st.write(f"• {rec}")
    
    # Export Results Display
    if result.get('export_results'):
        st.header("📤 Export Results")
        export_results = result['export_results']
        
        for format_type, export_info in export_results.items():
            if format_type != 'package' and export_info.get('status') == 'success':
                st.success(f"✓ {format_type.upper()} export completed successfully")
        
        # Package download if available
        if export_results.get('package'):
            st.info("Multiple format export package created!")
    
    # JSON Preview
    with st.expander("📄 JSON Structure Preview"):
        # Show first page as sample
        if result.get("pages"):
            sample_page = result["pages"][0]
            st.json(sample_page)
        else:
            st.json(result)
    
    # Create downloadable ZIP
    zip_path = create_download_package(result, temp_dir, original_filename)
    
    if zip_path and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            zip_data = f.read()
        
        filename = sanitize_filename(original_filename.replace('.pdf', ''))
        download_filename = f"{filename}_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        st.download_button(
            label="📥 Download Results (ZIP)",
            data=zip_data,
            file_name=download_filename,
            mime="application/zip",
            type="primary"
        )
    
    # Page-by-page results
    st.header("📖 Page-by-Page Results")
    
    if result.get("pages"):
        for i, page_data in enumerate(result["pages"]):
            with st.expander(f"Page {page_data.get('page_number', i+1)}"):
                
                # Text content
                if page_data.get("text"):
                    st.subheader("Text Content")
                    st.text_area(
                        "Extracted Text", 
                        page_data["text"][:1000] + "..." if len(page_data["text"]) > 1000 else page_data["text"],
                        height=150,
                        key=f"text_{i}"
                    )
                
                # Images
                if page_data.get("diagrams"):
                    st.subheader(f"Images/Diagrams ({len(page_data['diagrams'])})")
                    cols = st.columns(min(3, len(page_data["diagrams"])))
                    for j, diagram in enumerate(page_data["diagrams"]):
                        with cols[j % 3]:
                            image_path = os.path.join(temp_dir, "assets", "images", f"{diagram['diagram_id']}.png")
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"Image {j+1}", use_container_width=True)
                
                # Tables
                if page_data.get("tables"):
                    st.subheader(f"Tables ({len(page_data['tables'])})")
                    for j, table in enumerate(page_data["tables"]):
                        st.write(f"**Table {j+1}**")
                        # Try to display table data if available
                        json_path = os.path.join(temp_dir, "assets", "tables", f"{table['table_id']}.json")
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    table_data = json.load(f)
                                st.json(table_data)
                            except:
                                st.write("Table data available in download package")
                
                # Equations
                if page_data.get("equations"):
                    st.subheader(f"Equations ({len(page_data['equations'])})")
                    for j, equation in enumerate(page_data["equations"]):
                        st.write(f"**Equation {j+1}**: {equation.get('latex', 'N/A')}")

def create_download_package(result, temp_dir, original_filename):
    """Create a ZIP package with all extracted data"""
    
    try:
        filename = sanitize_filename(original_filename.replace('.pdf', ''))
        zip_path = os.path.join(temp_dir, f"{filename}_extraction.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add main JSON result
            json_path = os.path.join(temp_dir, "extraction_result.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            zipf.write(json_path, "extraction_result.json")
            
            # Add assets directory if it exists
            assets_dir = os.path.join(temp_dir, "assets")
            if os.path.exists(assets_dir):
                for root, dirs, files in os.walk(assets_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arc_path)
        
        return zip_path
    
    except Exception as e:
        st.error(f"Error creating download package: {str(e)}")
        return None

def display_batch_results(batch_results, temp_dir):
    """Display batch processing results"""
    
    st.success("Batch processing completed!")
    
    # Batch summary
    stats = batch_results['stats']
    summary = batch_results['batch_summary']
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", stats['total_files'])
    with col2:
        st.metric("Successfully Processed", stats['processed_files'])
    with col3:
        st.metric("Failed Files", stats['failed_files'])
    with col4:
        processing_time = stats.get('end_time', datetime.now()) - stats.get('start_time', datetime.now())
        st.metric("Processing Time", str(processing_time).split('.')[0])
    
    # Detailed metrics
    st.header("📊 Batch Processing Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", summary.get('total_pages_processed', 0))
    with col2:
        st.metric("Total Images", summary.get('total_images_extracted', 0))
    with col3:
        st.metric("Total Tables", summary.get('total_tables_detected', 0))
    with col4:
        st.metric("Total Equations", summary.get('total_equations_found', 0))
    
    # Quality overview for batch
    quality_scores = []
    for result in batch_results['results']:
        if result.get('quality_assessment'):
            quality_scores.append(result['quality_assessment']['overall_score'])
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        st.info(f"**Average Quality Score:** {avg_quality:.1f}%")
        
        # Quality distribution
        high_quality = len([score for score in quality_scores if score >= 80])
        medium_quality = len([score for score in quality_scores if 60 <= score < 80])
        low_quality = len([score for score in quality_scores if score < 60])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Quality (≥80%)", high_quality)
        with col2:
            st.metric("Medium Quality (60-80%)", medium_quality)
        with col3:
            st.metric("Low Quality (<60%)", low_quality)
    
    # Individual file results
    st.header("📄 Individual File Results")
    
    for i, result in enumerate(batch_results['results']):
        file_name = result.get('file_metadata', {}).get('original_filename', f'Document {i+1}')
        
        with st.expander(f"{file_name}"):
            # File metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pages", len(result.get('pages', [])))
            with col2:
                text_length = sum(len(page.get('text', '')) for page in result.get('pages', []))
                st.metric("Text Characters", f"{text_length:,}")
            with col3:
                images = sum(len(page.get('diagrams', [])) for page in result.get('pages', []))
                st.metric("Images", images)
            with col4:
                tables = sum(len(page.get('tables', [])) for page in result.get('pages', []))
                st.metric("Tables", tables)
            
            # Quality score if available
            if result.get('quality_assessment'):
                quality_score = result['quality_assessment']['overall_score']
                if quality_score >= 80:
                    st.success(f"Quality Score: {quality_score:.1f}%")
                elif quality_score >= 60:
                    st.warning(f"Quality Score: {quality_score:.1f}%")
                else:
                    st.error(f"Quality Score: {quality_score:.1f}%")
    
    # Batch download
    if batch_results.get('export_package'):
        st.header("📥 Download Batch Results")
        
        try:
            with open(batch_results['export_package'], 'rb') as f:
                zip_data = f.read()
            
            filename = f"batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            st.download_button(
                label="📦 Download Complete Batch Results",
                data=zip_data,
                file_name=filename,
                mime="application/zip",
                type="primary"
            )
            
            st.info("The batch results package includes all extracted data, quality reports, and export formats.")
            
        except Exception as e:
            st.error(f"Error preparing batch download: {str(e)}")

if __name__ == "__main__":
    main()
