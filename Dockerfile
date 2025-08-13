FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p image-requests

ENV PYTHONPATH=/app

WORKDIR /app/src/api

EXPOSE 8001

CMD ["python", "main.py"]