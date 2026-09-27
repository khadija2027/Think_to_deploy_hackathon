"""Daily ingestion: publish one aligned, immutable snapshot for the chat API."""
from datetime import timedelta
import pendulum
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

# No model loading, document scanning, or directory creation during DAG parsing.
from rag_core import pipeline

def discover_documents(**context):
    # A manual trigger may use {"force_rebuild": true} to regenerate
    # intermediate JSON artifacts even when the corpus fingerprint is unchanged.
    configuration = context["dag_run"].conf or {}
    force_rebuild = configuration.get("force_rebuild") is True
    work = pipeline.discover(context["run_id"], force=force_rebuild)
    if work is None:
        raise AirflowSkipException("No documents yet, or the published corpus is unchanged.")
    return work

with DAG(
    dag_id="safran_robust_faiss_rag_pipeline",
    description="Validate, parse, mask PII, chunk, embed and atomically publish HR documents",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "rag", "retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["hr", "rag", "faiss"],
) as dag:
    discover = PythonOperator(task_id="validate_and_discover", python_callable=discover_documents,
                              execution_timeout=timedelta(minutes=10))
    parse = PythonOperator(task_id="parse_documents", python_callable=pipeline.parse_documents,
                           op_args=[discover.output], execution_timeout=timedelta(minutes=30))
    anonymize = PythonOperator(task_id="anonymize_documents", python_callable=pipeline.anonymize_documents,
                               op_args=[parse.output], execution_timeout=timedelta(minutes=10))
    chunk = PythonOperator(task_id="chunk_documents", python_callable=pipeline.chunk_documents,
                           op_args=[anonymize.output], execution_timeout=timedelta(minutes=10))
    publish = PythonOperator(task_id="publish_index", python_callable=pipeline.publish_index,
                             op_args=[chunk.output], execution_timeout=timedelta(minutes=60))
    discover >> parse >> anonymize >> chunk >> publish
