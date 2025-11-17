# ===================================================================================
# Project: ChatTensorFlow
# File: src/services/indexing/opensearch_indexer.py
# Description: Indexes TensorFlow documentation chunks into OpenSearch for efficient retrieval
# Author: LALAN KUMAR
# Created: [09-11-2025]
# Updated: [09-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import OpenSearchException
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import ijson
import os
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple
from datetime import datetime, timezone

# Dynamically add project root to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logger import logging
from src.services.indexing.index_config import TENSORFLOW_INDEX_BODY


# STREAMING CHUNK READER
def stream_json_chunks(filepath: str | Path) -> Iterable[Dict[str, Any]]:
    """Stream huge JSON array file safely using ijson (no RAM explosion)."""
    logging.info(f"Streaming chunks from: {filepath}")

    with open(filepath, "rb") as f:
        for chunk in ijson.items(f, "item"):
            yield chunk


# PREPARE CHUNK WITH METADATA + VALIDATION
def prepare_chunk(chunk: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize + validate a chunk. Returns None if invalid."""
    
    if "embedding" not in chunk or not isinstance(chunk["embedding"], list):
        return None

    # Generate stable chunk_id if missing
    if "chunk_id" not in chunk:
        hash_input = (
            f"{chunk.get('source_url', '')}_"
            f"{chunk.get('heading', '')}_"
            f"{chunk.get('word_count', 0)}"
        )
        chunk["chunk_id"] = hashlib.md5(hash_input.encode()).hexdigest()[:16]

    # Inject timestamp
    chunk["indexed_at"] = datetime.now(timezone.utc).isoformat()

    # Enforce required fields
    defaults = {
        "page_type": "documentation",
        "heading": "",
        "text": "",
        "full_text": chunk.get("text", ""),
        "enriched_text": "",
        "page_title": "",
        "code_blocks": [],
        "has_code": False,
        "total_code_lines": 0,
        "source_url": "",
        "breadcrumbs": [],
        "word_count": 0,
    }
    for k, v in defaults.items():
        chunk.setdefault(k, v)

    return chunk


# MAIN INDEXING FUNCTION
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def index_chunks(
    chunks_embeddings_file: str | Path,
    index_name: str,
    client: OpenSearch,
    index_body: Dict[str, Any] = None,
    overwrite: bool = True,
    batch_size: int = 500
) -> Tuple[int, Dict[str, Any]]:
    """
    Stream + batch index huge JSON file into OpenSearch safely.
    """

    if index_body is None:
        index_body = TENSORFLOW_INDEX_BODY

    file_path = Path(chunks_embeddings_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")

    logging.info("=" * 70)
    logging.info("Starting streaming OpenSearch indexing")
    logging.info("=" * 70)
    logging.info(f"Input file: {file_path}")

    # Create / overwrite index
    index_exists = client.indices.exists(index=index_name)

    if index_exists and overwrite:
        logging.info(f"Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)

    if not index_exists or overwrite:
        logging.info(f"Creating new index: {index_name}")
        client.indices.create(index=index_name, body=index_body)

    # STREAM + BATCH INDEX
    stats = {
        "total_chunks": 0,
        "indexed": 0,
        "skipped": 0,
        "chunks_with_code": 0,
        "total_code_blocks": 0,
        "page_types": {}
    }

    batch_actions = []
    processed = 0

    for raw_chunk in stream_json_chunks(file_path):

        processed += 1
        stats["total_chunks"] += 1

        chunk = prepare_chunk(raw_chunk)
        if chunk is None:
            stats["skipped"] += 1
            continue

        # Collect statistics
        if chunk.get("has_code"):
            stats["chunks_with_code"] += 1
        stats["total_code_blocks"] += len(chunk.get("code_blocks", []))
        page_type = chunk.get("page_type", "unknown")
        stats["page_types"][page_type] = stats["page_types"].get(page_type, 0) + 1

        # Add to bulk batch
        batch_actions.append({
            "_index": index_name,
            "_id": chunk["chunk_id"],
            "_source": chunk
        })

        # Execute batch when full
        if len(batch_actions) >= batch_size:
            success, failed = helpers.bulk(
                client,
                batch_actions,
                chunk_size=batch_size,
                request_timeout=120,
                raise_on_error=False
            )
            stats["indexed"] += success
            batch_actions = []

            logging.info(f"Indexed so far: {stats['indexed']} (processed: {processed})")

    # Final batch flush
    if batch_actions:
        success, failed = helpers.bulk(
            client,
            batch_actions,
            chunk_size=batch_size,
            request_timeout=120,
            raise_on_error=False
        )
        stats["indexed"] += success

    # Refresh index
    client.indices.refresh(index=index_name)

    # FINAL INDEX STATS
    try:
        count = client.count(index=index_name)["count"]
        size_bytes = client.indices.stats(index=index_name)["_all"]["total"]["store"]["size_in_bytes"]
        stats["final_document_count"] = count
        stats["index_size_mb"] = round(size_bytes / (1024**2), 2)
    except Exception as e:
        logging.warning(f"Index stats unavailable: {e}")

    logging.info("=" * 70)
    logging.info("Indexing completed!")
    logging.info(f"   - Total processed: {stats['total_chunks']}")
    logging.info(f"   - Indexed:         {stats['indexed']}")
    logging.info(f"   - Skipped:         {stats['skipped']}")
    logging.info(f"   - Index size:      {stats.get('index_size_mb', 0)} MB")
    logging.info("=" * 70)

    return stats["indexed"], stats
