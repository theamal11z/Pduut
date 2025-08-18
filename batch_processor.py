import os
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf_processor import PDFProcessor
from utils import create_output_structure, sanitize_filename, ensure_directory
import threading

class BatchPDFProcessor:
    """Process multiple PDFs in batch with progress tracking"""
    
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'start_time': None,
            'end_time': None
        }
        self.progress_lock = threading.Lock()
    
    def process_multiple_pdfs(self, file_list, processing_options, progress_callback=None):
        """Process multiple PDF files in parallel"""
        
        self.processing_stats['total_files'] = len(file_list)
        self.processing_stats['start_time'] = datetime.now()
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {}
                for file_info in file_list:
                    future = executor.submit(
                        self._process_single_pdf,
                        file_info,
                        processing_options,
                        progress_callback
                    )
                    future_to_file[future] = file_info
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            with self.progress_lock:
                                self.processing_stats['processed_files'] += 1
                                if result.get('pdf_metadata', {}).get('pages'):
                                    self.processing_stats['total_pages'] += result['pdf_metadata']['pages']
                        else:
                            with self.progress_lock:
                                self.processing_stats['failed_files'] += 1
                        
                        # Update progress
                        if progress_callback:
                            completed = self.processing_stats['processed_files'] + self.processing_stats['failed_files']
                            progress = int((completed / self.processing_stats['total_files']) * 100)
                            progress_callback(
                                progress,
                                f"Processed {completed}/{self.processing_stats['total_files']} files"
                            )
                    
                    except Exception as e:
                        print(f"Error processing {file_info['name']}: {str(e)}")
                        with self.progress_lock:
                            self.processing_stats['failed_files'] += 1
            
            self.processing_stats['end_time'] = datetime.now()
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
        
        return {
            'results': results,
            'stats': self.processing_stats,
            'batch_summary': self._create_batch_summary(results)
        }
    
    def _process_single_pdf(self, file_info, processing_options, progress_callback=None):
        """Process a single PDF file"""
        
        try:
            # Create temporary directory for this file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                pdf_path = os.path.join(temp_dir, file_info['name'])
                with open(pdf_path, "wb") as f:
                    f.write(file_info['data'])
                
                # Initialize processor with options
                processor = PDFProcessor(
                    ocr_languages=processing_options.get('ocr_languages', ['eng']),
                    extract_images=processing_options.get('extract_images', True),
                    extract_tables=processing_options.get('extract_tables', True),
                    extract_equations=processing_options.get('extract_equations', True)
                )
                
                # Process the PDF
                result = processor.process_pdf(pdf_path, temp_dir)
                
                if result:
                    # Add file metadata
                    result['file_metadata'] = {
                        'original_filename': file_info['name'],
                        'file_size': file_info['size'],
                        'processing_time': datetime.now().isoformat()
                    }
                
                return result
                
        except Exception as e:
            print(f"Error processing single PDF {file_info['name']}: {str(e)}")
            return None
    
    def _create_batch_summary(self, results):
        """Create a summary of the batch processing results"""
        
        summary = {
            'total_documents': len(results),
            'total_pages_processed': sum(r.get('pdf_metadata', {}).get('pages', 0) for r in results),
            'total_text_characters': sum(
                sum(len(page.get('text', '')) for page in r.get('pages', []))
                for r in results
            ),
            'total_images_extracted': sum(
                sum(len(page.get('diagrams', [])) for page in r.get('pages', []))
                for r in results
            ),
            'total_tables_detected': sum(
                sum(len(page.get('tables', [])) for page in r.get('pages', []))
                for r in results
            ),
            'total_equations_found': sum(
                sum(len(page.get('equations', [])) for page in r.get('pages', []))
                for r in results
            ),
            'processing_time': self._calculate_processing_time(),
            'successful_files': len(results),
            'failed_files': self.processing_stats['failed_files']
        }
        
        return summary
    
    def _calculate_processing_time(self):
        """Calculate total processing time"""
        
        if self.processing_stats['start_time'] and self.processing_stats['end_time']:
            duration = self.processing_stats['end_time'] - self.processing_stats['start_time']
            return {
                'total_seconds': duration.total_seconds(),
                'formatted': str(duration).split('.')[0]  # Remove microseconds
            }
        return {'total_seconds': 0, 'formatted': '00:00:00'}
    
    def create_batch_export(self, batch_results, output_dir):
        """Create a comprehensive export package for batch results"""
        
        try:
            # Create main results directory
            batch_dir = os.path.join(output_dir, "batch_results")
            ensure_directory(batch_dir)
            
            # Create individual file directories
            for i, result in enumerate(batch_results['results']):
                file_name = result.get('file_metadata', {}).get('original_filename', f'document_{i}')
                safe_name = sanitize_filename(file_name.replace('.pdf', ''))
                
                file_dir = os.path.join(batch_dir, safe_name)
                ensure_directory(file_dir)
                
                # Save individual result
                with open(os.path.join(file_dir, 'extraction_result.json'), 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Create batch summary
            summary_path = os.path.join(batch_dir, 'batch_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'batch_stats': batch_results['stats'],
                    'batch_summary': batch_results['batch_summary'],
                    'file_list': [r.get('file_metadata', {}).get('original_filename', 'unknown') 
                                 for r in batch_results['results']]
                }, f, indent=2, ensure_ascii=False)
            
            # Create consolidated dataset
            consolidated_data = self._create_consolidated_dataset(batch_results['results'])
            consolidated_path = os.path.join(batch_dir, 'consolidated_dataset.json')
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
            
            # Create ZIP archive
            zip_path = os.path.join(output_dir, f"batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(batch_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arc_path)
            
            return zip_path
            
        except Exception as e:
            print(f"Error creating batch export: {str(e)}")
            return None
    
    def _create_consolidated_dataset(self, results):
        """Create a consolidated dataset from all processed documents"""
        
        consolidated = {
            'dataset_metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_documents': len(results),
                'total_pages': sum(len(r.get('pages', [])) for r in results),
                'dataset_type': 'educational_pdf_extraction'
            },
            'documents': [],
            'global_statistics': {
                'total_text_length': 0,
                'total_images': 0,
                'total_tables': 0,
                'total_equations': 0,
                'language_distribution': {},
                'content_type_distribution': {}
            }
        }
        
        for result in results:
            doc_data = {
                'document_id': sanitize_filename(result.get('file_metadata', {}).get('original_filename', 'unknown')),
                'metadata': result.get('pdf_metadata', {}),
                'pages': result.get('pages', []),
                'extraction_info': result.get('extraction_info', {})
            }
            consolidated['documents'].append(doc_data)
            
            # Update global statistics
            stats = consolidated['global_statistics']
            stats['total_text_length'] += sum(len(page.get('text', '')) for page in result.get('pages', []))
            stats['total_images'] += sum(len(page.get('diagrams', [])) for page in result.get('pages', []))
            stats['total_tables'] += sum(len(page.get('tables', [])) for page in result.get('pages', []))
            stats['total_equations'] += sum(len(page.get('equations', [])) for page in result.get('pages', []))
        
        return consolidated

class QualityAssessment:
    """Assess the quality of extracted data for RAG applications"""
    
    def __init__(self):
        self.quality_thresholds = {
            'text_confidence': 70,
            'min_text_length': 50,
            'image_resolution': (100, 100),
            'table_min_cells': 4
        }
    
    def assess_extraction_quality(self, extraction_result):
        """Assess the overall quality of extraction results"""
        
        assessment = {
            'overall_score': 0,
            'text_quality': 0,
            'image_quality': 0,
            'table_quality': 0,
            'equation_quality': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            pages = extraction_result.get('pages', [])
            
            if not pages:
                assessment['issues'].append('No pages extracted')
                return assessment
            
            # Assess text quality
            text_scores = []
            for page in pages:
                text = page.get('text', '')
                if len(text) >= self.quality_thresholds['min_text_length']:
                    # Simple quality heuristics
                    word_count = len(text.split())
                    char_variety = len(set(text.lower()))
                    score = min(100, (word_count * char_variety) / 100)
                    text_scores.append(score)
            
            assessment['text_quality'] = np.mean(text_scores) if text_scores else 0
            
            # Assess image quality
            image_scores = []
            total_images = 0
            for page in pages:
                diagrams = page.get('diagrams', [])
                total_images += len(diagrams)
                for diagram in diagrams:
                    width = diagram.get('width', 0)
                    height = diagram.get('height', 0)
                    if width >= self.quality_thresholds['image_resolution'][0] and \
                       height >= self.quality_thresholds['image_resolution'][1]:
                        image_scores.append(80)  # Good resolution
                    else:
                        image_scores.append(40)  # Low resolution
            
            assessment['image_quality'] = np.mean(image_scores) if image_scores else 0
            
            # Assess table quality
            table_scores = []
            for page in pages:
                tables = page.get('tables', [])
                for table in tables:
                    rows = table.get('rows', 0)
                    cols = table.get('cols', 0)
                    if rows * cols >= self.quality_thresholds['table_min_cells']:
                        table_scores.append(75)
                    else:
                        table_scores.append(25)
            
            assessment['table_quality'] = np.mean(table_scores) if table_scores else 0
            
            # Assess equation quality
            equation_scores = []
            for page in pages:
                equations = page.get('equations', [])
                for equation in equations:
                    confidence = equation.get('confidence', 0)
                    if confidence >= 0.6:
                        equation_scores.append(confidence * 100)
                    else:
                        equation_scores.append(confidence * 50)
            
            assessment['equation_quality'] = np.mean(equation_scores) if equation_scores else 0
            
            # Calculate overall score
            quality_weights = {
                'text': 0.4,
                'image': 0.2,
                'table': 0.2,
                'equation': 0.2
            }
            
            assessment['overall_score'] = (
                assessment['text_quality'] * quality_weights['text'] +
                assessment['image_quality'] * quality_weights['image'] +
                assessment['table_quality'] * quality_weights['table'] +
                assessment['equation_quality'] * quality_weights['equation']
            )
            
            # Generate recommendations
            if assessment['text_quality'] < 50:
                assessment['recommendations'].append('Consider using higher resolution scans for better text extraction')
            
            if assessment['image_quality'] < 50 and total_images > 0:
                assessment['recommendations'].append('Image quality is low - consider adjusting extraction parameters')
            
            if assessment['overall_score'] < 60:
                assessment['issues'].append('Overall extraction quality is below recommended threshold for RAG applications')
            
        except Exception as e:
            assessment['issues'].append(f'Error during quality assessment: {str(e)}')
        
        return assessment