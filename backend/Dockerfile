FROM python:3.11-slim

WORKDIR /app

# Install required system libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app code
COPY . .

# Run the app with a single Gunicorn worker to reduce memory usage
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "app:app"]
