FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p logs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]