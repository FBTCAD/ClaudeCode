# DeepSeek-OCR PDF Reader - Usage Guide

This guide explains how to use the DeepSeek-OCR PDF Reader to extract text from PDF files and generate section-by-section summaries.

## Overview

The `deepseek_ocr_pdf_reader.py` script uses the DeepSeek-OCR model from Hugging Face to:
- Convert PDF pages to images
- Extract text from each page using state-of-the-art OCR
- Generate summaries for each page/section
- Save results in JSON format for further processing

## Installation

### 1. Install Dependencies

#### Basic Installation (CPU)
```bash
pip install -r requirements.txt
```

#### GPU Installation (CUDA)
For GPU acceleration with flash attention:
```bash
pip install -r requirements.txt
pip install flash-attn>=2.7.3
```

Note: flash-attn requires CUDA 11.8+ and may take some time to compile.

### 2. Install System Dependencies

For PDF processing, you also need `poppler-utils`:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download and install from: https://github.com/oschwartz10612/poppler-windows/releases

## Usage

### Command Line Interface

#### Basic Usage
```bash
python deepseek_ocr_pdf_reader.py your_document.pdf
```

This will:
- Process all pages in the PDF
- Extract text using OCR
- Generate summaries for each page
- Save results to `your_document_ocr_results.json`

#### Advanced Options

**Specify output file:**
```bash
python deepseek_ocr_pdf_reader.py document.pdf --output results.json
```

**Set image quality (DPI):**
```bash
python deepseek_ocr_pdf_reader.py document.pdf --dpi 300
```
Higher DPI = better quality but slower processing. Default is 200.

**Disable summary generation:**
```bash
python deepseek_ocr_pdf_reader.py document.pdf --no-summary
```

**Force CPU mode:**
```bash
python deepseek_ocr_pdf_reader.py document.pdf --device cpu
```

**Force GPU mode:**
```bash
python deepseek_ocr_pdf_reader.py document.pdf --device cuda
```

### Python API Usage

You can also use the script as a Python module:

```python
from deepseek_ocr_pdf_reader import DeepSeekOCRPDFReader

# Initialize the reader
reader = DeepSeekOCRPDFReader(device='cuda')  # or 'cpu'

# Process a PDF
results = reader.process_pdf(
    pdf_path='document.pdf',
    output_path='results.json',
    dpi=200,
    generate_summary=True
)

# Access results
for page in results['pages']:
    print(f"Page {page['page_number']}:")
    print(f"Text: {page['text'][:100]}...")
    print(f"Summary: {page['summary']}")
    print()
```

### Processing Individual Images

```python
from deepseek_ocr_pdf_reader import DeepSeekOCRPDFReader
from PIL import Image

# Initialize reader
reader = DeepSeekOCRPDFReader()

# Load and process an image
image = Image.open('document_page.png')
text = reader.process_image(image)
print(text)

# Generate summary
summary = reader.summarize_text(text)
print(summary)
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "pdf_path": "document.pdf",
  "total_pages": 3,
  "pages": [
    {
      "page_number": 1,
      "text": "Full extracted text from page 1...",
      "summary": "Brief summary of page 1 content..."
    },
    {
      "page_number": 2,
      "text": "Full extracted text from page 2...",
      "summary": "Brief summary of page 2 content..."
    }
  ]
}
```

## Performance Tips

### GPU Acceleration
- The script automatically uses GPU if available
- Flash attention requires CUDA 11.8+ and flash-attn package
- GPU processing is significantly faster (10-50x depending on hardware)

### Memory Management
- Large PDFs may require significant RAM
- Process pages in smaller batches if you encounter memory issues
- Reduce DPI if memory is limited (try 150 or 100)

### Processing Speed
- **CPU**: ~10-30 seconds per page (depending on complexity)
- **GPU**: ~1-5 seconds per page (with flash attention)
- Higher DPI increases processing time proportionally

## Example Workflow

### 1. Process a research paper
```bash
python deepseek_ocr_pdf_reader.py research_paper.pdf --dpi 300 --output paper_analysis.json
```

### 2. Quick scan of a document
```bash
python deepseek_ocr_pdf_reader.py document.pdf --dpi 150 --no-summary
```

### 3. Batch processing multiple PDFs
```python
from deepseek_ocr_pdf_reader import DeepSeekOCRPDFReader
import glob

reader = DeepSeekOCRPDFReader()

for pdf_file in glob.glob('*.pdf'):
    print(f"Processing {pdf_file}...")
    results = reader.process_pdf(
        pdf_path=pdf_file,
        output_path=f"{pdf_file[:-4]}_results.json"
    )
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Try CPU mode instead
python deepseek_ocr_pdf_reader.py document.pdf --device cpu
```

**2. PDF conversion fails**
```
Error: poppler-utils not installed
```
Solution: Install poppler-utils (see Installation section)

**3. Flash attention not available**
The script will automatically fall back to standard attention. This is normal on systems without CUDA or flash-attn.

**4. Model download issues**
The first run will download the model (~several GB). Ensure you have:
- Stable internet connection
- Sufficient disk space (~10GB free)
- Hugging Face Hub access

### Memory Requirements

- **Minimum**: 8GB RAM (CPU mode, low DPI)
- **Recommended**: 16GB RAM (CPU mode, high DPI)
- **GPU**: 8GB VRAM minimum (flash attention)

## Model Information

- **Model**: deepseek-ai/DeepSeek-OCR
- **Architecture**: Vision-Language Model optimized for OCR
- **License**: Check Hugging Face model card
- **Size**: ~7GB
- **Language Support**: Primarily English and Chinese

## Advanced Customization

### Custom OCR Prompts

You can customize the OCR prompt for specific tasks:

```python
reader = DeepSeekOCRPDFReader()
image = Image.open('form.png')

# Extract tables
text = reader.process_image(
    image,
    prompt="<image>\n<|grounding|>Extract all tables from this document."
)

# Extract specific information
text = reader.process_image(
    image,
    prompt="<image>\n<|grounding|>Extract all dates and names from this document."
)
```

### Custom Summarization

The built-in summarization is extractive (first N sentences). For better summaries:

```python
# Use a dedicated summarization model
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Process with DeepSeek-OCR
reader = DeepSeekOCRPDFReader()
text = reader.process_image(image)

# Generate better summary
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

## License

This script is provided as-is. Please check the DeepSeek-OCR model license on Hugging Face for model usage terms.

## Support

For issues related to:
- **This script**: Open an issue in this repository
- **DeepSeek-OCR model**: Visit https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Dependencies**: Check respective package documentation
