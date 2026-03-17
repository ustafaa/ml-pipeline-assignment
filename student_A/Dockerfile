FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY student_a_gan.py .
COPY mnist_test.csv .

CMD ["python", "student_a_gan.py"]