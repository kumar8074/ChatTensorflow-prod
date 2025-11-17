# ===================================================================================
# Project: ChatTensorFlow
# File: airflow/dags/tf_ingestion.py
# Description: Main Airflow DAG for TensorFlow documentation ingestion
# Author: LALAN KUMAR
# Created: [11-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.1.0
# ===================================================================================

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum
import sys

# Ensure dags can import ingestion_tasks
sys.path.insert(0, '/opt/airflow/dags')

from ingestion_tasks.scrapper_task import scrape_tensorflow_urls
from ingestion_tasks.chunking_task import chunk_tensorflow_content
from ingestion_tasks.embedding_task import generate_tensorflow_embeddings_task
from ingestion_tasks.indexing_task import index_tensorflow_to_opensearch

# Define IST timezone
ist = pendulum.timezone('Asia/Kolkata')

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tensorflow_documentation_ingestion',
    default_args=default_args,
    description='Ingest TensorFlow documentation into OpenSearch with embeddings',
    schedule_interval='0 23 27 * *',  # Monthly, 11:00 PM IST on the 27th (cron)
    start_date=datetime(2025, 11, 1, tzinfo=ist),
    catchup=False,
    tags=['tensorflow', 'documentation', 'rag', 'opensearch', 'monthly'],
) as dag:

    crawl_urls = PythonOperator(
        task_id='crawl_tensorflow_urls',
        python_callable=scrape_tensorflow_urls,
        op_kwargs={
            'output_dir': 'temp',
            'output_file': 'docs_rag.json'
        }
    )

    chunk_content = PythonOperator(
        task_id='chunk_tensorflow_content',
        python_callable=chunk_tensorflow_content,
        op_kwargs={
            'input_file': 'temp/docs_rag.json',
            'output_dir': 'temp/chunked_data',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_pages': None
        }
    )

    generate_embeddings_task = PythonOperator(
        task_id='generate_tensorflow_embeddings',
        python_callable=generate_tensorflow_embeddings_task,
        op_kwargs={
            'chunks_file': 'temp/chunked_data/chunks_for_rag.jsonl',
            'output_file': 'temp/chunked_data/chunks_with_embeddings.json',
            'batch_size': 100
        }
    )

    index_to_opensearch = PythonOperator(
        task_id='index_to_opensearch',
        python_callable=index_tensorflow_to_opensearch,
        op_kwargs={
            'embedded_chunks_file': 'temp/chunked_data/chunks_with_embeddings.json',
            'index_name': 'tensorflow_documentation'
        }
    )

    # DAG dependencies
    crawl_urls >> chunk_content >> generate_embeddings_task >> index_to_opensearch
