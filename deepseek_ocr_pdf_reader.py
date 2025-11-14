"""
DeepSeek-OCR PDF Reader with Section-by-Section Summarization

This script uses the DeepSeek-OCR model from Hugging Face to:
1. Load and process PDF files
2. Extract text from each page using OCR
3. Generate summaries for each section

Requirements:
- torch>=2.6.0
- transformers>=4.46.3
- tokenizers>=0.20.3
- Pillow
- pdf2image
- einops
- addict
- easydict
- flash-attn>=2.7.3 (for GPU acceleration)
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from pdf2image import convert_from_path
from typing import List, Dict, Tuple
import json
from pathlib import Path


class DeepSeekOCRPDFReader:
    """
    A class to handle OCR processing of PDF files using DeepSeek-OCR model.
    """

    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR', device: str = None):
        """
        Initialize the DeepSeek-OCR model and tokenizer.

        Args:
            model_name: Name of the model on Hugging Face
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing DeepSeek-OCR on {self.device}...")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        print("Loading model (this may take a while)...")
        if self.device == 'cuda':
            # Use flash attention for GPU
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
                self.model = self.model.eval().cuda().to(torch.bfloat16)
                print("Model loaded with flash_attention_2")
            except Exception as e:
                print(f"Flash attention not available, using standard attention: {e}")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                self.model = self.model.eval().cuda()
        else:
            # CPU mode
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True
            )
            self.model = self.model.eval()

        print("Model loaded successfully!")

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (higher = better quality but slower)

        Returns:
            List of PIL Image objects, one per page
        """
        print(f"Converting PDF to images (DPI={dpi})...")
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            print(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            raise

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process a single image with OCR.

        Args:
            image: PIL Image object
            prompt: Custom prompt for the model (default: convert to markdown)

        Returns:
            Extracted text from the image
        """
        if prompt is None:
            # Default prompt to convert document to markdown
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                images=[image],
                return_tensors="pt"
            )

            # Move to device
            if self.device == 'cuda':
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False
                )

            # Decode output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return result

        except Exception as e:
            print(f"Error processing image: {e}")
            return f"[Error processing image: {e}]"

    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the extracted text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Summary of the text
        """
        # For now, we'll create a simple extractive summary
        # In production, you might want to use a dedicated summarization model

        if not text or len(text.strip()) == 0:
            return "[No content to summarize]"

        # Split into sentences (simple approach)
        sentences = text.replace('\n', ' ').split('. ')

        # Take first few sentences as summary
        summary_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence) + 2  # +2 for '. '
            else:
                break

        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.'):
            summary += '...'

        return summary if summary else text[:max_length] + "..."

    def process_pdf(
        self,
        pdf_path: str,
        output_path: str = None,
        dpi: int = 200,
        generate_summary: bool = True
    ) -> Dict:
        """
        Process an entire PDF file with OCR and optional summarization.

        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the results (JSON format)
            dpi: DPI for PDF to image conversion
            generate_summary: Whether to generate summaries for each page

        Returns:
            Dictionary containing OCR results and summaries
        """
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*60}\n")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi)

        # Process each page
        results = {
            'pdf_path': pdf_path,
            'total_pages': len(images),
            'pages': []
        }

        for i, image in enumerate(images, 1):
            print(f"\nProcessing page {i}/{len(images)}...")

            # Extract text with OCR
            text = self.process_image(image)

            page_result = {
                'page_number': i,
                'text': text
            }

            # Generate summary if requested
            if generate_summary:
                print(f"Generating summary for page {i}...")
                summary = self.summarize_text(text)
                page_result['summary'] = summary

                # Print summary to console
                print(f"\n--- Page {i} Summary ---")
                print(summary)
                print(f"{'─'*40}\n")

            results['pages'].append(page_result)

        # Save results if output path provided
        if output_path:
            print(f"\nSaving results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("Results saved successfully!")

        print(f"\n{'='*60}")
        print(f"Processing complete! Processed {len(images)} pages.")
        print(f"{'='*60}\n")

        return results


def main():
    """
    Main function to demonstrate usage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Process PDF files with DeepSeek-OCR and generate summaries'
    )
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to the PDF file to process'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to save the JSON results (default: <pdf_name>_ocr_results.json)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI for PDF to image conversion (default: 200)'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Disable summary generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return

    # Set default output path
    if args.output is None:
        pdf_name = Path(args.pdf_path).stem
        args.output = f"{pdf_name}_ocr_results.json"

    # Initialize reader
    try:
        reader = DeepSeekOCRPDFReader(device=args.device)
    except Exception as e:
        print(f"Error initializing DeepSeek-OCR: {e}")
        print("\nMake sure you have installed all required dependencies:")
        print("pip install torch transformers tokenizers Pillow pdf2image einops addict easydict")
        print("\nFor GPU support, also install: pip install flash-attn")
        return

    # Process PDF
    try:
        results = reader.process_pdf(
            pdf_path=args.pdf_path,
            output_path=args.output,
            dpi=args.dpi,
            generate_summary=not args.no_summary
        )

        print(f"\n✓ Successfully processed {results['total_pages']} pages")
        print(f"✓ Results saved to: {args.output}")

    except Exception as e:
        print(f"\nError processing PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
