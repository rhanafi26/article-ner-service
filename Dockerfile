# Gunakan base image Python yang ringan
FROM python:3.11-slim

# Set working directory di dalam kontainer
WORKDIR /app

# Salin file kebutuhan dependensi
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin folder aplikasi dan model
COPY ./app ./app
COPY ./models ./models
COPY ./src ./src

# Expose port yang digunakan FastAPI
EXPOSE 8000

# Perintah untuk menjalankan server FastAPI saat kontainer di-start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
