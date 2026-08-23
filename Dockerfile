FROM python:3.11-slim

# Install dependensi sistem dasar yang sering dibutuhkan Python
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command khusus untuk Railway (menggunakan variabel $PORT dari Railway)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
