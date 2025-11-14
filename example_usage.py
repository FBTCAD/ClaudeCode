"""
Example usage of DeepSeek-OCR PDF Reader

This script demonstrates various ways to use the DeepSeekOCRPDFReader class.
"""

from deepseek_ocr_pdf_reader import DeepSeekOCRPDFReader
from PIL import Image
import os


def example_1_basic_pdf_processing():
    """
    Example 1: Basic PDF processing with summaries
    """
    print("\n" + "="*60)
    print("Example 1: Basic PDF Processing")
    print("="*60)

    # Initialize the reader
    reader = DeepSeekOCRPDFReader(device='cuda')  # Use 'cpu' if no GPU

    # Process a PDF file
    pdf_path = 'sample_document.pdf'

    if not os.path.exists(pdf_path):
        print(f"Note: {pdf_path} not found. This is just an example.")
        print("Replace with your actual PDF path.")
        return

    results = reader.process_pdf(
        pdf_path=pdf_path,
        output_path='results.json',
        dpi=200,
        generate_summary=True
    )

    # Access the results
    print(f"\nProcessed {results['total_pages']} pages")
    for page in results['pages']:
        print(f"\nPage {page['page_number']} Summary:")
        print(page['summary'])


def example_2_process_single_image():
    """
    Example 2: Process a single image
    """
    print("\n" + "="*60)
    print("Example 2: Process Single Image")
    print("="*60)

    # Initialize the reader
    reader = DeepSeekOCRPDFReader()

    # Load an image
    image_path = 'document_page.png'

    if not os.path.exists(image_path):
        print(f"Note: {image_path} not found. This is just an example.")
        print("Replace with your actual image path.")
        return

    image = Image.open(image_path)

    # Extract text
    text = reader.process_image(image)
    print("\nExtracted text:")
    print(text)

    # Generate summary
    summary = reader.summarize_text(text, max_length=150)
    print("\nSummary:")
    print(summary)


def example_3_batch_processing():
    """
    Example 3: Batch process multiple PDFs
    """
    print("\n" + "="*60)
    print("Example 3: Batch Processing Multiple PDFs")
    print("="*60)

    import glob

    # Initialize once for all PDFs
    reader = DeepSeekOCRPDFReader()

    # Find all PDF files in current directory
    pdf_files = glob.glob('*.pdf')

    if not pdf_files:
        print("No PDF files found in current directory.")
        print("This is just an example of batch processing.")
        return

    print(f"Found {len(pdf_files)} PDF files")

    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file}...")

        output_file = f"{os.path.splitext(pdf_file)[0]}_ocr_results.json"

        try:
            results = reader.process_pdf(
                pdf_path=pdf_file,
                output_path=output_file,
                dpi=150,  # Lower DPI for faster batch processing
                generate_summary=True
            )
            print(f"✓ Completed: {results['total_pages']} pages")
        except Exception as e:
            print(f"✗ Error processing {pdf_file}: {e}")


def example_4_custom_prompt():
    """
    Example 4: Using custom prompts for specific tasks
    """
    print("\n" + "="*60)
    print("Example 4: Custom Prompts")
    print("="*60)

    reader = DeepSeekOCRPDFReader()

    image_path = 'table_document.png'

    if not os.path.exists(image_path):
        print(f"Note: {image_path} not found. This is just an example.")
        return

    image = Image.open(image_path)

    # Extract tables specifically
    print("\n1. Extracting tables...")
    table_text = reader.process_image(
        image,
        prompt="<image>\n<|grounding|>Extract all tables from this document and format them in markdown."
    )
    print(table_text)

    # Extract specific information
    print("\n2. Extracting dates and names...")
    info_text = reader.process_image(
        image,
        prompt="<image>\n<|grounding|>Extract all dates and person names from this document."
    )
    print(info_text)


def example_5_low_memory_mode():
    """
    Example 5: Process large PDFs with memory constraints
    """
    print("\n" + "="*60)
    print("Example 5: Memory-Efficient Processing")
    print("="*60)

    reader = DeepSeekOCRPDFReader(device='cpu')  # CPU uses less memory

    pdf_path = 'large_document.pdf'

    if not os.path.exists(pdf_path):
        print(f"Note: {pdf_path} not found. This is just an example.")
        return

    # Use lower DPI to reduce memory usage
    results = reader.process_pdf(
        pdf_path=pdf_path,
        output_path='results.json',
        dpi=100,  # Lower DPI = less memory
        generate_summary=False  # Skip summaries to save time
    )

    print(f"Processed {results['total_pages']} pages efficiently")


def example_6_only_specific_pages():
    """
    Example 6: Process only specific pages from a PDF
    """
    print("\n" + "="*60)
    print("Example 6: Process Specific Pages")
    print("="*60)

    reader = DeepSeekOCRPDFReader()

    pdf_path = 'document.pdf'

    if not os.path.exists(pdf_path):
        print(f"Note: {pdf_path} not found. This is just an example.")
        return

    # Convert PDF to images
    images = reader.pdf_to_images(pdf_path)

    # Process only pages 1, 3, and 5
    pages_to_process = [0, 2, 4]  # 0-indexed

    results = {
        'pdf_path': pdf_path,
        'total_pages': len(pages_to_process),
        'pages': []
    }

    for idx in pages_to_process:
        if idx < len(images):
            print(f"\nProcessing page {idx + 1}...")
            text = reader.process_image(images[idx])
            summary = reader.summarize_text(text)

            results['pages'].append({
                'page_number': idx + 1,
                'text': text,
                'summary': summary
            })

    print(f"\nProcessed {len(results['pages'])} selected pages")


def main():
    """
    Main function to run examples
    """
    import sys

    print("\n" + "="*60)
    print("DeepSeek-OCR PDF Reader - Usage Examples")
    print("="*60)

    examples = {
        '1': ('Basic PDF Processing', example_1_basic_pdf_processing),
        '2': ('Process Single Image', example_2_process_single_image),
        '3': ('Batch Processing', example_3_batch_processing),
        '4': ('Custom Prompts', example_4_custom_prompt),
        '5': ('Low Memory Mode', example_5_low_memory_mode),
        '6': ('Specific Pages Only', example_6_only_specific_pages),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter example number (or 'all' to run all): ").strip()

    if choice.lower() == 'all':
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    elif choice in examples:
        _, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")
        return

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
