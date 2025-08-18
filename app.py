import streamlit as st
import os
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from pdf_processor import PDFProcessor
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
    
    # File upload
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
            process_pdf(
                uploaded_file, 
                ocr_languages, 
                extract_images, 
                extract_tables, 
                extract_equations
            )

def process_pdf(uploaded_file, ocr_languages, extract_images, extract_tables, extract_equations):
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
                extract_equations=extract_equations
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
                
                # Display results
                display_results(result, temp_dir, uploaded_file.name)
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
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
    col1, col2, col3, col4 = st.columns(4)
    
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
                                st.image(image_path, caption=f"Image {j+1}", use_column_width=True)
                
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

if __name__ == "__main__":
    main()
