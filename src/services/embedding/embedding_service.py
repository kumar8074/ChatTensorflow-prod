# ===================================================================================
# Project: ChatTensorFlow
# File: src/services/embeddings/embedding_service.py
# Description: Embeds and stores TensorFlow documentation chunks with embeddings
# Author: LALAN KUMAR
# Created: [09-11-2025]
# Updated: [09-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from langchain.embeddings import Embeddings

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logger import logging
from src.config import EMBEDDING_MODEL

def load_tensorflow_chunks(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load TensorFlow chunks from JSONL (JSON Lines) file into a list of dictionaries.

    Args:
        file_path (str | Path): Path to the `.jsonl` file containing TensorFlow chunks.

    Returns:
        List[Dict[str, Any]]: A list of parsed chunk objects (each line is one dict).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON at line {line_number}: {e.msg}")
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_number}: {e.msg}", e.doc, e.pos
                ) from e
    
    logging.info(f"Loaded {len(chunks)} TensorFlow chunks from {file_path}")
    
    # Log some statistics
    chunks_with_code = sum(1 for c in chunks if c.get('has_code', False))
    page_types = {}
    for chunk in chunks:
        pt = chunk.get('page_type', 'unknown')
        page_types[pt] = page_types.get(pt, 0) + 1
    
    logging.info(f"   - Chunks with code: {chunks_with_code}")
    logging.info(f"   - Page types: {dict(sorted(page_types.items(), key=lambda x: x[1], reverse=True))}")
    
    return chunks


def generate_tensorflow_embeddings(
        chunks: List[Dict[str, Any]], 
        batch_size: int = 100, 
        client: Embeddings | None = None,
        show_progress: bool = True
    ) -> List[List[float]]:
    """
    Generate embeddings for TensorFlow documentation chunks using a LangChain-compatible embedding model.

    Args:
        chunks (List[Dict[str, Any]]): A list of TensorFlow chunk dictionaries.
        batch_size (int, optional): Number of texts to process per batch. Defaults to 100.
        client (Embeddings | None, optional): A LangChain embedding model instance.
        show_progress (bool, optional): Whether to show progress logs. Defaults to True.

    Returns:
        List[List[float]]: A list of embedding vectors (one per chunk).

    Raises:
        ValueError: If `client` is not provided or chunks are invalid.
    """
    if client is None:
        raise ValueError("An embedding client must be provided (e.g., GoogleGenerativeAIEmbeddings).")
    
    if not chunks:
        raise ValueError("Chunks list is empty. Please provide valid chunks.")
    
    logging.info(f"Using Embedding model: {client.__class__.__name__}")
    logging.info(f"Total chunks to embed: {len(chunks)}")
    logging.info(f"Batch size: {batch_size}")

    embeddings: List[List[float]] = []

    # Use enriched_text for embeddings (contains full context for better retrieval)
    texts = []
    for idx, chunk in enumerate(chunks):
        if 'enriched_text' not in chunk:
            logging.warning(f"Chunk {idx} missing 'enriched_text', using 'full_text' as fallback")
            text = chunk.get('full_text') or chunk.get('text', '')
        else:
            text = chunk['enriched_text']
        
        if not text or not text.strip():
            logging.warning(f"Chunk {idx} has empty text, using placeholder")
            text = f"Empty chunk from {chunk.get('source_url', 'unknown')}"
        
        texts.append(text)

    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx:batch_idx + batch_size]
        current_batch_num = (batch_idx // batch_size) + 1
        
        try:
            batch_embeddings: List[List[float]] = client.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = batch_idx + len(batch)
                percentage = (progress / len(texts)) * 100
                logging.info(f"Progress: [{current_batch_num}/{total_batches}] {progress}/{len(texts)} chunks ({percentage:.1f}%)")
        
        except Exception as e:
            logging.error(f"Error processing batch {current_batch_num}: {e}")
            # Add empty embeddings for this batch to maintain alignment
            empty_embedding = [0.0] * 768  # Common embedding dimension
            embeddings.extend([empty_embedding] * len(batch))
            logging.warning(f"Added {len(batch)} empty embeddings as placeholders")

    logging.info(f"Generated {len(embeddings)} embeddings successfully")
    
    return embeddings


def combine_tensorflow_chunks_embeddings(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    output_file_path: str | Path
) -> None:
    """
    Combine TensorFlow chunks with their embeddings and save to disk as JSON.

    Args:
        chunks (List[Dict[str, Any]]): List of TensorFlow chunk dictionaries.
        embeddings (List[List[float]]): List of embedding vectors corresponding to chunks.
        output_file_path (str | Path): File path where the combined JSON data will be saved.

    Returns:
        None

    Raises:
        ValueError: If the lengths of `chunks` and `embeddings` do not match.
        OSError: If there is an issue writing to the output file.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same."
        )

    logging.info("Combining embeddings with chunks...")

    # Combine embeddings with chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logging.info("Embeddings combined with all chunks")

    # Ensure parent directory exists
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embedded chunks to JSON file
    logging.info(f"Saving to: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Chunks with embeddings saved successfully")
    logging.info(f"   - File size: {file_size_mb:.2f} MB")
    logging.info(f"   - Total chunks: {len(chunks)}")



# Example usage:
if __name__ == "__main__":
    
    start_time = time.time()
    
    logging.info("="*60)
    logging.info("Starting TensorFlow embedding generation process...")
    logging.info("="*60)
    
    # Configuration
    INPUT_FILE = "temp/chunked_data/chunks_for_rag.jsonl"
    OUTPUT_FILE = "temp/chunked_data/chunks_with_embeddings.json"
    BATCH_SIZE = 100
    
    try:
        # Step 1: Load chunks
        logging.info("\nStep 1: Loading TensorFlow chunks...")
        chunks = load_tensorflow_chunks(INPUT_FILE)
        
        # Step 2: Generate embeddings
        logging.info("\nStep 2: Generating embeddings...")
        embeddings = generate_tensorflow_embeddings(
            chunks=chunks, 
            batch_size=BATCH_SIZE, 
            client=EMBEDDING_MODEL
        )
        
        logging.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 3: Combine and save
        logging.info("Step 3: Combining chunks with embeddings and saving...")
        combine_tensorflow_chunks_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            output_file_path=OUTPUT_FILE
        )
        
        duration = time.time() - start_time
        
        logging.info("\n" + "="*60)
        logging.info("Embedding generation process completed successfully!")
        
        logging.info(f"Total time: {duration:.2f} seconds")
        logging.info(f"Average time per chunk: {(duration/len(chunks)):.3f} seconds")
        logging.info("="*60)
        
        logging.info("\nNext steps:")
        logging.info("1. Load the embeddings into your vector database (e.g., Pinecone, Chroma)")
        logging.info("2. Create similarity search indices")
        logging.info("3. Test retrieval with sample queries")
        logging.info("4. Integrate with your RAG pipeline")
        
    except Exception as e:
        logging.error(f"\nError during embedding generation: {e}")
        import traceback
        traceback.print_exc()
        raise