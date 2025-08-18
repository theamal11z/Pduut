import os
import re
import json
from datetime import datetime

def create_output_structure(metadata, pages_data):
    """Create the final output structure for the PDF extraction"""
    
    return {
        "pdf_metadata": metadata,
        "pages": pages_data,
        "extraction_info": {
            "total_pages_processed": len(pages_data),
            "extraction_timestamp": datetime.now().isoformat(),
            "total_text_length": sum(len(page.get("text", "")) for page in pages_data),
            "total_images": sum(len(page.get("diagrams", [])) for page in pages_data),
            "total_tables": sum(len(page.get("tables", [])) for page in pages_data),
            "total_equations": sum(len(page.get("equations", [])) for page in pages_data)
        }
    }

def sanitize_filename(filename):
    """Sanitize filename by removing/replacing invalid characters"""
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove extra spaces and replace with underscore
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized

def ensure_directory(directory_path):
    """Ensure directory exists, create if it doesn't"""
    
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_text_content(text):
    """Clean and normalize text content"""
    
    if not text:
        return ""
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Fix common OCR errors (basic fixes)
    replacements = {
        'rn': 'm',  # Common OCR confusion
        '|': 'l',   # Pipe to lowercase L
        '0': 'O',   # Zero to O in contexts where it makes sense
    }
    
    # Apply replacements cautiously
    for old, new in replacements.items():
        # Only replace in specific contexts to avoid over-correction
        pass  # Placeholder for more sophisticated text cleaning
    
    return cleaned

def extract_key_phrases(text, max_phrases=10):
    """Extract key phrases from text for indexing"""
    
    if not text:
        return []
    
    # Simple keyword extraction (in production, use more sophisticated NLP)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'must'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    top_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [phrase[0] for phrase in top_phrases[:max_phrases]]

def generate_page_summary(page_data):
    """Generate a summary for a page based on its content"""
    
    summary = {
        "page_number": page_data.get("page_number", 0),
        "has_text": bool(page_data.get("text", "").strip()),
        "text_length": len(page_data.get("text", "")),
        "image_count": len(page_data.get("diagrams", [])),
        "table_count": len(page_data.get("tables", [])),
        "equation_count": len(page_data.get("equations", [])),
        "key_phrases": [],
        "content_type": "unknown"
    }
    
    # Extract key phrases from text
    if summary["has_text"]:
        summary["key_phrases"] = extract_key_phrases(page_data["text"])
    
    # Determine content type
    if summary["equation_count"] > 2:
        summary["content_type"] = "mathematics_heavy"
    elif summary["table_count"] > 0:
        summary["content_type"] = "data_tables"
    elif summary["image_count"] > 2:
        summary["content_type"] = "diagram_heavy"
    elif summary["text_length"] > 1000:
        summary["content_type"] = "text_heavy"
    elif summary["text_length"] > 100:
        summary["content_type"] = "mixed_content"
    else:
        summary["content_type"] = "minimal_content"
    
    return summary

def validate_extraction_result(result):
    """Validate the extraction result structure"""
    
    errors = []
    warnings = []
    
    # Check required fields
    if not result.get("pdf_metadata"):
        errors.append("Missing pdf_metadata")
    
    if not result.get("pages"):
        errors.append("Missing pages data")
    elif not isinstance(result["pages"], list):
        errors.append("Pages data must be a list")
    
    # Validate pages
    for i, page in enumerate(result.get("pages", [])):
        if not isinstance(page, dict):
            errors.append(f"Page {i} must be a dictionary")
            continue
        
        if not page.get("page_number"):
            warnings.append(f"Page {i} missing page_number")
        
        # Check for empty content
        has_content = any([
            page.get("text", "").strip(),
            page.get("diagrams"),
            page.get("tables"),
            page.get("equations")
        ])
        
        if not has_content:
            warnings.append(f"Page {i} appears to have no extracted content")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "page_count": len(result.get("pages", [])),
        "has_content": any(page.get("text", "").strip() for page in result.get("pages", []))
    }

def create_extraction_report(result):
    """Create a detailed extraction report"""
    
    validation = validate_extraction_result(result)
    
    report = {
        "extraction_summary": {
            "status": "success" if validation["is_valid"] else "failed",
            "total_pages": len(result.get("pages", [])),
            "pages_with_text": sum(1 for page in result.get("pages", []) if page.get("text", "").strip()),
            "total_images": sum(len(page.get("diagrams", [])) for page in result.get("pages", [])),
            "total_tables": sum(len(page.get("tables", [])) for page in result.get("pages", [])),
            "total_equations": sum(len(page.get("equations", [])) for page in result.get("pages", [])),
            "total_text_characters": sum(len(page.get("text", "")) for page in result.get("pages", []))
        },
        "validation_results": validation,
        "page_summaries": [generate_page_summary(page) for page in result.get("pages", [])],
        "extraction_timestamp": datetime.now().isoformat()
    }
    
    return report
