import json
import csv
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from utils import sanitize_filename, ensure_directory
import zipfile
import markdown
import yaml

class ExportManager:
    """Handle multiple export formats for extracted data"""
    
    def __init__(self):
        self.supported_formats = [
            'json', 'csv', 'xml', 'markdown', 'yaml', 'html'
        ]
    
    def export_data(self, extraction_result, output_dir, formats=['json'], include_assets=True):
        """Export extraction results in multiple formats"""
        
        export_results = {}
        
        for format_type in formats:
            if format_type not in self.supported_formats:
                print(f"Unsupported format: {format_type}")
                continue
            
            try:
                if format_type == 'json':
                    result = self._export_json(extraction_result, output_dir)
                elif format_type == 'csv':
                    result = self._export_csv(extraction_result, output_dir)
                elif format_type == 'xml':
                    result = self._export_xml(extraction_result, output_dir)
                elif format_type == 'markdown':
                    result = self._export_markdown(extraction_result, output_dir)
                elif format_type == 'yaml':
                    result = self._export_yaml(extraction_result, output_dir)
                elif format_type == 'html':
                    result = self._export_html(extraction_result, output_dir)
                
                export_results[format_type] = result
                
            except Exception as e:
                print(f"Error exporting to {format_type}: {str(e)}")
                export_results[format_type] = {'error': str(e)}
        
        # Create comprehensive export package
        if len(export_results) > 1:
            package_path = self._create_export_package(extraction_result, export_results, output_dir, include_assets)
            export_results['package'] = package_path
        
        return export_results
    
    def _export_json(self, extraction_result, output_dir):
        """Export as JSON with different structure options"""
        
        try:
            # Standard JSON export
            json_path = os.path.join(output_dir, 'extraction_result.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            # Flat structure for RAG systems
            flat_path = os.path.join(output_dir, 'flat_structure.json')
            flat_data = self._create_flat_structure(extraction_result)
            with open(flat_path, 'w', encoding='utf-8') as f:
                json.dump(flat_data, f, indent=2, ensure_ascii=False)
            
            # Page-wise JSON files
            pages_dir = os.path.join(output_dir, 'pages_json')
            ensure_directory(pages_dir)
            
            for page in extraction_result.get('pages', []):
                page_num = page.get('page_number', 1)
                page_path = os.path.join(pages_dir, f'page_{page_num}.json')
                with open(page_path, 'w', encoding='utf-8') as f:
                    json.dump(page, f, indent=2, ensure_ascii=False)
            
            return {
                'main_file': json_path,
                'flat_structure': flat_path,
                'pages_directory': pages_dir,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _export_csv(self, extraction_result, output_dir):
        """Export structured data as CSV files"""
        
        try:
            csv_dir = os.path.join(output_dir, 'csv_export')
            ensure_directory(csv_dir)
            
            # Page-level summary CSV
            pages_csv = os.path.join(csv_dir, 'pages_summary.csv')
            with open(pages_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Text_Length', 'Image_Count', 'Table_Count', 'Equation_Count'])
                
                for page in extraction_result.get('pages', []):
                    writer.writerow([
                        page.get('page_number', 0),
                        len(page.get('text', '')),
                        len(page.get('diagrams', [])),
                        len(page.get('tables', [])),
                        len(page.get('equations', []))
                    ])
            
            # Text content CSV
            text_csv = os.path.join(csv_dir, 'text_content.csv')
            with open(text_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Text_Content'])
                
                for page in extraction_result.get('pages', []):
                    writer.writerow([
                        page.get('page_number', 0),
                        page.get('text', '').replace('\n', ' ')
                    ])
            
            # Images CSV
            images_csv = os.path.join(csv_dir, 'images.csv')
            with open(images_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Image_ID', 'Description', 'Width', 'Height', 'Image_Link'])
                
                for page in extraction_result.get('pages', []):
                    for diagram in page.get('diagrams', []):
                        writer.writerow([
                            page.get('page_number', 0),
                            diagram.get('diagram_id', ''),
                            diagram.get('description', ''),
                            diagram.get('width', 0),
                            diagram.get('height', 0),
                            diagram.get('image_link', '')
                        ])
            
            # Equations CSV
            equations_csv = os.path.join(csv_dir, 'equations.csv')
            with open(equations_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Equation_ID', 'LaTeX', 'Original_Text', 'Confidence'])
                
                for page in extraction_result.get('pages', []):
                    for equation in page.get('equations', []):
                        writer.writerow([
                            page.get('page_number', 0),
                            equation.get('equation_id', ''),
                            equation.get('latex', ''),
                            equation.get('original_text', ''),
                            equation.get('confidence', 0)
                        ])
            
            return {
                'directory': csv_dir,
                'files': ['pages_summary.csv', 'text_content.csv', 'images.csv', 'equations.csv'],
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _export_xml(self, extraction_result, output_dir):
        """Export as structured XML"""
        
        try:
            xml_path = os.path.join(output_dir, 'extraction_result.xml')
            
            # Create root element
            root = ET.Element('pdf_extraction')
            
            # Add metadata
            metadata_elem = ET.SubElement(root, 'metadata')
            pdf_meta = extraction_result.get('pdf_metadata', {})
            for key, value in pdf_meta.items():
                meta_elem = ET.SubElement(metadata_elem, key)
                meta_elem.text = str(value)
            
            # Add pages
            pages_elem = ET.SubElement(root, 'pages')
            for page_data in extraction_result.get('pages', []):
                page_elem = ET.SubElement(pages_elem, 'page')
                page_elem.set('number', str(page_data.get('page_number', 0)))
                
                # Text content
                if page_data.get('text'):
                    text_elem = ET.SubElement(page_elem, 'text')
                    text_elem.text = page_data['text']
                
                # Images
                if page_data.get('diagrams'):
                    images_elem = ET.SubElement(page_elem, 'images')
                    for diagram in page_data['diagrams']:
                        img_elem = ET.SubElement(images_elem, 'image')
                        img_elem.set('id', diagram.get('diagram_id', ''))
                        img_elem.set('link', diagram.get('image_link', ''))
                        desc_elem = ET.SubElement(img_elem, 'description')
                        desc_elem.text = diagram.get('description', '')
                
                # Tables
                if page_data.get('tables'):
                    tables_elem = ET.SubElement(page_elem, 'tables')
                    for table in page_data['tables']:
                        table_elem = ET.SubElement(tables_elem, 'table')
                        table_elem.set('id', table.get('table_id', ''))
                        table_elem.set('csv_link', table.get('csv_link', ''))
                        table_elem.set('json_link', table.get('json_link', ''))
                
                # Equations
                if page_data.get('equations'):
                    equations_elem = ET.SubElement(page_elem, 'equations')
                    for equation in page_data['equations']:
                        eq_elem = ET.SubElement(equations_elem, 'equation')
                        eq_elem.set('id', equation.get('equation_id', ''))
                        latex_elem = ET.SubElement(eq_elem, 'latex')
                        latex_elem.text = equation.get('latex', '')
            
            # Write XML file
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            return {
                'file_path': xml_path,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _export_markdown(self, extraction_result, output_dir):
        """Export as Markdown documentation"""
        
        try:
            md_path = os.path.join(output_dir, 'extraction_result.md')
            
            with open(md_path, 'w', encoding='utf-8') as f:
                # Title and metadata
                pdf_meta = extraction_result.get('pdf_metadata', {})
                f.write(f"# {pdf_meta.get('title', 'PDF Extraction Results')}\n\n")
                f.write(f"**Author:** {pdf_meta.get('author', 'Unknown')}\n")
                f.write(f"**Pages:** {pdf_meta.get('pages', 0)}\n")
                f.write(f"**Processed:** {pdf_meta.get('processed_date', 'Unknown')}\n\n")
                
                # Table of contents
                f.write("## Table of Contents\n\n")
                for page_data in extraction_result.get('pages', []):
                    page_num = page_data.get('page_number', 0)
                    f.write(f"- [Page {page_num}](#page-{page_num})\n")
                f.write("\n")
                
                # Page content
                for page_data in extraction_result.get('pages', []):
                    page_num = page_data.get('page_number', 0)
                    f.write(f"## Page {page_num}\n\n")
                    
                    # Text content
                    if page_data.get('text'):
                        f.write("### Text Content\n\n")
                        f.write(page_data['text'])
                        f.write("\n\n")
                    
                    # Images
                    if page_data.get('diagrams'):
                        f.write("### Images and Diagrams\n\n")
                        for i, diagram in enumerate(page_data['diagrams'], 1):
                            f.write(f"#### Image {i}: {diagram.get('description', 'Diagram')}\n\n")
                            if diagram.get('image_link'):
                                f.write(f"![{diagram.get('description', 'Image')}]({diagram['image_link']})\n\n")
                    
                    # Tables
                    if page_data.get('tables'):
                        f.write("### Tables\n\n")
                        for i, table in enumerate(page_data['tables'], 1):
                            f.write(f"#### Table {i}\n\n")
                            f.write(f"- Rows: {table.get('rows', 0)}\n")
                            f.write(f"- Columns: {table.get('cols', 0)}\n")
                            if table.get('csv_link'):
                                f.write(f"- [Download CSV]({table['csv_link']})\n")
                            f.write("\n")
                    
                    # Equations
                    if page_data.get('equations'):
                        f.write("### Mathematical Equations\n\n")
                        for i, equation in enumerate(page_data['equations'], 1):
                            f.write(f"#### Equation {i}\n\n")
                            f.write(f"```latex\n{equation.get('latex', '')}\n```\n\n")
            
            return {
                'file_path': md_path,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _export_yaml(self, extraction_result, output_dir):
        """Export as YAML configuration"""
        
        try:
            yaml_path = os.path.join(output_dir, 'extraction_result.yaml')
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(extraction_result, f, default_flow_style=False, allow_unicode=True)
            
            return {
                'file_path': yaml_path,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _export_html(self, extraction_result, output_dir):
        """Export as HTML documentation"""
        
        try:
            html_path = os.path.join(output_dir, 'extraction_result.html')
            
            html_content = self._generate_html_content(extraction_result)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'file_path': html_path,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _generate_html_content(self, extraction_result):
        """Generate HTML content from extraction results"""
        
        pdf_meta = extraction_result.get('pdf_metadata', {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pdf_meta.get('title', 'PDF Extraction Results')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .page {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
        .page-header {{ background: #e9e9e9; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }}
        .equation {{ background: #fff9c4; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; }}
        .image {{ margin: 10px 0; text-align: center; }}
        .table-info {{ background: #e6f3ff; padding: 10px; margin: 10px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>{pdf_meta.get('title', 'PDF Extraction Results')}</h1>
    
    <div class="metadata">
        <h2>Document Information</h2>
        <p><strong>Author:</strong> {pdf_meta.get('author', 'Unknown')}</p>
        <p><strong>Pages:</strong> {pdf_meta.get('pages', 0)}</p>
        <p><strong>Processed:</strong> {pdf_meta.get('processed_date', 'Unknown')}</p>
    </div>
"""
        
        # Add pages
        for page_data in extraction_result.get('pages', []):
            page_num = page_data.get('page_number', 0)
            html += f"""
    <div class="page">
        <div class="page-header">
            <h2>Page {page_num}</h2>
        </div>
"""
            
            # Text content
            if page_data.get('text'):
                html += f"""
        <h3>Text Content</h3>
        <div class="text-content">
            <pre>{page_data['text']}</pre>
        </div>
"""
            
            # Images
            if page_data.get('diagrams'):
                html += "<h3>Images and Diagrams</h3>\n"
                for diagram in page_data['diagrams']:
                    html += f"""
        <div class="image">
            <h4>{diagram.get('description', 'Diagram')}</h4>
            <img src="{diagram.get('image_link', '')}" alt="{diagram.get('description', 'Image')}" style="max-width: 100%; height: auto;">
        </div>
"""
            
            # Tables
            if page_data.get('tables'):
                html += "<h3>Tables</h3>\n"
                for table in page_data['tables']:
                    html += f"""
        <div class="table-info">
            <h4>Table: {table.get('table_id', '')}</h4>
            <p>Dimensions: {table.get('rows', 0)} rows × {table.get('cols', 0)} columns</p>
            <p><a href="{table.get('csv_link', '')}">Download CSV</a> | <a href="{table.get('json_link', '')}">Download JSON</a></p>
        </div>
"""
            
            # Equations
            if page_data.get('equations'):
                html += "<h3>Mathematical Equations</h3>\n"
                for equation in page_data['equations']:
                    html += f"""
        <div class="equation">
            <strong>LaTeX:</strong> {equation.get('latex', '')}<br>
            <strong>Original:</strong> {equation.get('original_text', '')}<br>
            <strong>Confidence:</strong> {equation.get('confidence', 0):.2f}
        </div>
"""
            
            html += "</div>\n"
        
        html += """
</body>
</html>
"""
        
        return html
    
    def _create_flat_structure(self, extraction_result):
        """Create a flat structure optimized for RAG systems"""
        
        flat_data = []
        
        for page_data in extraction_result.get('pages', []):
            page_num = page_data.get('page_number', 0)
            
            # Text chunks
            if page_data.get('text'):
                flat_data.append({
                    'id': f"page_{page_num}_text",
                    'type': 'text',
                    'page': page_num,
                    'content': page_data['text'],
                    'metadata': {
                        'length': len(page_data['text']),
                        'source': 'ocr'
                    }
                })
            
            # Image descriptions
            for i, diagram in enumerate(page_data.get('diagrams', [])):
                flat_data.append({
                    'id': f"page_{page_num}_image_{i}",
                    'type': 'image_description',
                    'page': page_num,
                    'content': diagram.get('description', ''),
                    'metadata': {
                        'image_link': diagram.get('image_link', ''),
                        'width': diagram.get('width', 0),
                        'height': diagram.get('height', 0)
                    }
                })
            
            # Equations
            for i, equation in enumerate(page_data.get('equations', [])):
                flat_data.append({
                    'id': f"page_{page_num}_equation_{i}",
                    'type': 'equation',
                    'page': page_num,
                    'content': equation.get('latex', ''),
                    'metadata': {
                        'original_text': equation.get('original_text', ''),
                        'confidence': equation.get('confidence', 0)
                    }
                })
            
            # Tables
            for i, table in enumerate(page_data.get('tables', [])):
                flat_data.append({
                    'id': f"page_{page_num}_table_{i}",
                    'type': 'table',
                    'page': page_num,
                    'content': f"Table with {table.get('rows', 0)} rows and {table.get('cols', 0)} columns",
                    'metadata': {
                        'csv_link': table.get('csv_link', ''),
                        'json_link': table.get('json_link', ''),
                        'rows': table.get('rows', 0),
                        'cols': table.get('cols', 0)
                    }
                })
        
        return {
            'dataset_info': {
                'total_chunks': len(flat_data),
                'source_document': extraction_result.get('pdf_metadata', {}).get('title', 'Unknown'),
                'creation_date': datetime.now().isoformat()
            },
            'chunks': flat_data
        }
    
    def _create_export_package(self, extraction_result, export_results, output_dir, include_assets):
        """Create a comprehensive export package"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            package_name = f"complete_extraction_{timestamp}"
            package_path = os.path.join(output_dir, f"{package_name}.zip")
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all export files
                for format_type, result in export_results.items():
                    if format_type == 'package':
                        continue
                    
                    if result.get('status') == 'success':
                        if 'file_path' in result:
                            zipf.write(result['file_path'], os.path.basename(result['file_path']))
                        elif 'directory' in result:
                            # Add directory contents
                            for root, dirs, files in os.walk(result['directory']):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, output_dir)
                                    zipf.write(file_path, arc_name)
                
                # Add assets if requested
                if include_assets:
                    assets_dir = os.path.join(output_dir, 'assets')
                    if os.path.exists(assets_dir):
                        for root, dirs, files in os.walk(assets_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arc_name = os.path.relpath(file_path, output_dir)
                                zipf.write(file_path, arc_name)
                
                # Add README
                readme_content = self._generate_readme(extraction_result, export_results)
                zipf.writestr('README.md', readme_content)
            
            return package_path
            
        except Exception as e:
            print(f"Error creating export package: {str(e)}")
            return None
    
    def _generate_readme(self, extraction_result, export_results):
        """Generate README for the export package"""
        
        pdf_meta = extraction_result.get('pdf_metadata', {})
        
        readme = f"""# PDF Extraction Results Package

## Document Information
- **Title:** {pdf_meta.get('title', 'Unknown')}
- **Author:** {pdf_meta.get('author', 'Unknown')}
- **Pages:** {pdf_meta.get('pages', 0)}
- **Processing Date:** {pdf_meta.get('processed_date', 'Unknown')}

## Package Contents

"""
        
        for format_type, result in export_results.items():
            if format_type == 'package':
                continue
            
            if result.get('status') == 'success':
                readme += f"### {format_type.upper()} Export\n"
                if 'file_path' in result:
                    readme += f"- File: `{os.path.basename(result['file_path'])}`\n"
                elif 'files' in result:
                    readme += f"- Directory: `{os.path.basename(result['directory'])}/`\n"
                    for file in result['files']:
                        readme += f"  - `{file}`\n"
                readme += "\n"
        
        readme += """## Usage Instructions

### For RAG Systems
- Use `flat_structure.json` for easy integration with vector databases
- Text chunks are pre-segmented by page and content type
- Each chunk includes metadata for filtering and ranking

### For Analysis
- Use CSV files for statistical analysis
- XML/JSON for programmatic access
- Markdown/HTML for human-readable documentation

### Assets
- Images are stored in `assets/images/`
- Tables are available as both CSV and JSON in `assets/tables/`
- Text files are in `assets/text/`

## Technical Details
- OCR Engine: Tesseract with advanced preprocessing
- Image Processing: OpenCV-based detection and extraction
- Table Detection: Computer vision-based structure analysis
- Equation Recognition: Pattern matching with LaTeX conversion

Generated by PDUUT - PDF Data Extraction Tool
"""
        
        return readme