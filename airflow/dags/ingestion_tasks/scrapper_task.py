# ===================================================================================
# Project: ChatTensorFlow
# File: airflow/dags/ingestion_tasks/scraper_task.py
# Description: Airflow TASK to scrape TensorFlow documentation
# Author: LALAN KUMAR
# Created: [11-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.1.0
# ===================================================================================

import asyncio
import os
import sys
import traceback

# Ensure src is importable when Airflow runs the task
sys.path.insert(0, '/opt/airflow')

from src.services.scrapper.tensorflow_scrapper import crawl_tensorflow_docs
from src.logger import logging


def scrape_tensorflow_urls(
    sitemap_url: str | None = None,
    output_dir: str = "temp",
    output_file: str = "docs_rag.json",
    max_pages: int | None = None
) -> dict:
    """
    Wrapper task to run the TensorFlow sitemap-based scraper and write combined JSON.
    Uses the project's `crawl_tensorflow_docs` coroutine which writes to:
        os.path.join(SCRAPER_OUTPUT_DIR, SCRAPER_OUTPUT_FILE)
    This wrapper ensures the function can be called from Airflow's PythonOperator.
    """
    try:
        logging.info("Starting TensorFlow sitemap scraper task")

        # The scraper in src/services/scrapper handles its own output paths from config.
        # If you prefer passing the paths explicitly, adjust the module to accept op args.
        asyncio.run(crawl_tensorflow_docs())

        result = {
            "status": "success",
            "output_file": f"{output_dir}/{output_file}"
        }
        logging.info(f"Scraper finished: {result}")
        return result

    except Exception as e:
        logging.error(f"Scraper task failed: {e}")
        traceback.print_exc()
        raise
