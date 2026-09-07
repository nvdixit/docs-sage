
from pathlib import Path

from pypdf import PdfReader
from docx import Document
import ollama
from pymongo import MongoClient, MongoClient
import time
import base64

from preprocessing.html_text_extractor import HTMLTextExtractor

MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4"


def summarize(text=None, image=None, model_name=OLLAMA_MODEL, max_chars=5000):
    """Send text or an image to a locally hosted Ollama model for a short summary."""
    if text is None and image is None:
        raise ValueError("Either text or image must be provided for summarization.")

    if image is not None:
        # For multimodal models, the image should be passed as a base64-encoded string
        # inside the `images` field of the chat request.
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image and summarize its contents briefly.",
                    "images": [image],
                }
            ],
        )
        return response["message"]["content"]

    # Text-only path for regular documents.
    trimmed_text = text.strip()
    if len(trimmed_text) > max_chars:
        trimmed_text = trimmed_text[:max_chars] + "..."

    # Create a prompt that instructs the model to summarize the document's purpose and key points.
    summarization_prompt = (
        "Summarize the purpose and key points of the following document in a few sentences: "
        f"{trimmed_text}"
    )

    # Query the model with the prompt and return the generated summary.
    response = ollama.generate(
        model=model_name,
        prompt=summarization_prompt,
    )
    return response["response"]


def ingest_docs():
    '''
    Ingests documents from the "data" directory and returns a list of dictionaries containing file names and their content.
    Supported file types: .txt, .pdf, .docx, .doc, .html, .jpg, .jpeg, .png
    '''

    # Build the absolute path to the repository root so the script can reliably find the data folder.
    repo_root = Path(__file__).resolve().parent.parent
    # Use the repo root to locate the data directory where source documents live.
    data_dir = repo_root / "data"

    # If the data folder does not exist, return an empty list instead of crashing.
    if not data_dir.exists():
        return

    # Iterate through each file in the directory in a stable, sorted order.
    for file_path in sorted(data_dir.iterdir()):
        # Ignore directories and process only actual files.
        if not file_path.is_file():
            continue

        # Initialize content for the current file.
        content = ""

        try:
            # Process files based on their extension.

            if file_path.suffix == ".txt":
                # Read plain text files directly as UTF-8 text.
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            elif file_path.suffix == ".pdf":
                # Use pypdf to open PDF files and extract text from each page.
                with open(file_path, "rb") as f:
                    content = ""
                    # PdfReader reads the PDF file and exposes each page for extraction.
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        # Concatenate text from each page into a single string.
                        content += text or ""

            elif file_path.suffix == ".docx" or file_path.suffix == ".doc":
                # Use python-docx to read Word documents and extract text from paragraphs.
                try:
                    doc = Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    print("Warning: python-docx is not installed. Skipping Word document processing.")
                    continue

            elif file_path.suffix == ".html":
                # Read the raw HTML file as text. Ignore encoding errors to keep ingestion resilient.
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                # Feed the HTML to the custom parser to strip tags and keep visible content.
                parser = HTMLTextExtractor()
                parser.feed(html)
                content = parser.get_text()

            elif file_path.suffix == ".jpg" or file_path.suffix == ".jpeg" or file_path.suffix == ".png":
                # Images are not yet OCR-processed in this script, so leave the content empty.
                content = base64.b64encode(file_path.read_bytes()).decode('utf-8')
                
        except Exception:
            print(f"Warning: Failed to read or parse {file_path.name}. Skipping this file.")
            # Skip any file that fails to read or parse so the script continues processing others.
            continue

        # Create a summary for each successfully extracted document using the local Ollama model.
        summary = None

        # If the file is an image, we have gemma4 summarize the image
        if file_path.suffix in [".jpg", ".jpeg", ".png"]:
            summary = summarize(image=content)
        else:
            summary = summarize(text=content)

        # Save the filename, raw content, and AI summary for each document.
        processed_doc = {
            "file_name": file_path.name,
            "content": content,
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        # Insert the processed document into the MongoDB collection for later retrieval and analysis.
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client["docs-sage"]
        docs_sage_collection = db["processed-data"]
        docs_sage_collection.insert_one(processed_doc)


if __name__ == "__main__":
    # Run the ingestion script with "python3.12 -m preprocessing.ingest_docs"
    ingest_docs()