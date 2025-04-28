import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def text_to_embeddings(text, model):
    embeddings = model.encode(text)
    return embeddings.tolist()  # Chroma expects list

def process_folder_to_chroma(pdf_folder, chroma_collection_name="pdf_embeddings", model_name='all-MiniLM-L6-v2'):
    # Correct way to create Chroma client now
    client = chromadb.PersistentClient(path="./chroma_store")

    # Create or get the collection
    try:
        collection = client.get_collection(name=chroma_collection_name)
    except:
        collection = client.create_collection(name=chroma_collection_name)

    model = SentenceTransformer(model_name)

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {filename}...")

            # Step 1: Extract text
            extracted_text = pdf_to_text(pdf_path)

            # Step 2: Generate embeddings
            vector_embeddings = text_to_embeddings(extracted_text, model)

            # Step 3: Add to Chroma collection
            collection.add(
                documents=[extracted_text],
                embeddings=[vector_embeddings],
                metadatas=[{"filename": filename}],
                ids=[filename]  # Using filename as ID (must be unique)
            )

    print(f"All embeddings stored successfully in ./chroma_store/")

# Example usage
pdf_folder = "Docs"
process_folder_to_chroma(pdf_folder)
