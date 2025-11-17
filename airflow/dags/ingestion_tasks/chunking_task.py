# ===================================================================================
# Project: ChatTensorFlow
# File: airflow/dags/ingestion_tasks/chunking_task.py
# Description: Airflow TASK to load and intelligently chunk TensorFlow documentation
# Author: LALAN KUMAR
# Created: [11-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.1.0
# ===================================================================================

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/opt/airflow')

from src.services.chunking.content_chunker import TensorFlowContentChunker
from src.logger import logging


def chunk_tensorflow_content(
    input_file: str = "temp/docs_rag.json",
    output_dir: str = "temp/chunked_data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_pages: int | None = None
) -> dict:
    """
    Load scraped TensorFlow docs (temp/docs_rag.json) and produce chunked outputs:
      - temp/chunked_data/chunks_for_rag.jsonl
      - temp/chunked_data/all_chunks.json
      - temp/chunked_data/docs_with_chunks.json
      - temp/chunked_data/chunking_statistics.json
    """
    try:
        logging.info(f"Starting TensorFlow content chunking from: {input_file}")
        start_time = time.time()

        loader = TensorFlowContentChunker(
            input_file=input_file,
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        loader.process_all_documents()
        loader.save_data()

        duration = time.time() - start_time

        stats = {
            "status": "success",
            "total_documents": len(loader.processed_docs),
            "total_chunks": len(loader.all_chunks),
            "duration_seconds": round(duration, 2),
            "chunks_file": f"{output_dir}/chunks_for_rag.jsonl",
            "all_chunks_file": f"{output_dir}/all_chunks.json",
        }
        logging.info(f"Chunking finished: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Chunking task failed: {e}")
        raise
