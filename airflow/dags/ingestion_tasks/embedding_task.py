# ===================================================================================
# Project: ChatTensorFlow
# File: airflow/dags/ingestion_tasks/embedding_task.py
# Description: Airflow TASK to generate embeddings for TensorFlow chunks
# Author: LALAN KUMAR
# Created: [11-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.1.0
# ===================================================================================

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/opt/airflow')

from src.services.embedding.embedding_service import (
    load_tensorflow_chunks,
    generate_tensorflow_embeddings,
    combine_tensorflow_chunks_embeddings
)
from src.config import EMBEDDING_MODEL
from src.logger import logging


def generate_tensorflow_embeddings_task(
    chunks_file: str = "temp/chunked_data/chunks_for_rag.jsonl",
    output_file: str = "temp/chunked_data/chunks_with_embeddings.json",
    batch_size: int = 100
) -> dict:
    """
    1. Load chunks (JSONL)
    2. Generate embeddings (uses EMBEDDING_MODEL from src.config)
    3. Combine embeddings and save to output_file (JSON)
    """
    try:
        logging.info(f"Loading chunks from: {chunks_file}")
        chunks = load_tensorflow_chunks(chunks_file)
        logging.info(f"Loaded {len(chunks)} chunks")

        logging.info(f"Generating embeddings using model: {EMBEDDING_MODEL}")
        embeddings = generate_tensorflow_embeddings(
            chunks=chunks,
            batch_size=batch_size,
            client=EMBEDDING_MODEL
        )

        logging.info(f"Combining and saving embeddings to: {output_file}")
        combine_tensorflow_chunks_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            output_file_path=Path(output_file)
        )

        stats = {
            "status": "success",
            "total_chunks": len(chunks),
            "output_file": output_file
        }
        logging.info(f"Embedding generation finished: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise
