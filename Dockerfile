FROM python:3.11-slim AS api
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /requirements.txt
COPY rag_core /opt/project/rag_core
COPY frontend_chatbot /opt/project/frontend_chatbot
ENV PYTHONPATH=/opt/project
EXPOSE 8000
CMD ["uvicorn", "rag_core.web:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM api AS evaluation
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /evaluation
ENTRYPOINT ["python", "-u", "/evaluation/run_ragas.py"]
CMD []

FROM apache/airflow:2.8.1-python3.11 AS airflow
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 poppler-utils tesseract-ocr tesseract-ocr-fra antiword \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/airflow/pipeline /opt/airflow/index /opt/airflow/model-cache \
    && chown -R airflow:root /opt/airflow/pipeline /opt/airflow/index /opt/airflow/model-cache
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "apache-airflow==2.8.1" -r /requirements.txt
COPY rag_core /opt/project/rag_core
COPY init_airflow.py /opt/project/init_airflow.py
ENV PYTHONPATH=/opt/project
