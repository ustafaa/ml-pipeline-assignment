FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY check_threshold.py .

RUN echo "Simulating model download for Run ID: ${RUN_ID}"

CMD ["python", "train.py"]
