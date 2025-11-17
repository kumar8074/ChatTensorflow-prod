# ===================================================================================
# Project: ChatTensorFlow
# File: airflow/dags/ingestion_tasks/indexing_task.py
# Description: Airflow TASK to index TensorFlow chunks into OpenSearch
# Author: LALAN KUMAR
# Created: [11-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.1.0
# ===================================================================================

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/opt/airflow')

from src.services.indexing.opensearch_indexer import index_chunks
from src.services.indexing.index_config import TENSORFLOW_INDEX_BODY
from src.services.opensearch.factory import connect_to_opensearch
from src.config import OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS, INDEX_NAME
from src.logger import logging


def index_tensorflow_to_opensearch(
    embedded_chunks_file: str = "temp/chunked_data/chunks_with_embeddings.json",
    index_name: str | None = None
) -> dict:
    """
    Stream + batch index TensorFlow chunk+embeddings JSON into OpenSearch.
    """
    try:
        index_name = index_name or INDEX_NAME
        logging.info(f"Connecting to OpenSearch at {OPENSEARCH_HOST}")
        client, health = connect_to_opensearch(OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS)
        logging.info(f"OpenSearch cluster health: {health.get('status')}")

        logging.info(f"Indexing chunks from {embedded_chunks_file} into index {index_name}")
        total_indexed = index_chunks(
            chunks_embeddings_file=Path(embedded_chunks_file),
            index_name=index_name,
            client=client,
            index_body=TENSORFLOW_INDEX_BODY,
            overwrite=True,
            batch_size=500
        )

        stats = {
            "status": "success",
            "total_indexed": total_indexed,
            "index_name": index_name
        }
        logging.info(f"Indexing finished: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Indexing failed: {e}")
        raise
