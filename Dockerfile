FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with timeout for large packages
RUN pip install --no-cache-dir --timeout=1000 xgboost==1.7.6
RUN pip install --no-cache-dir --prefer-binary --timeout=300 -r requirements.txt

# Copy ALL application code
COPY . .

# Expose port
EXPOSE 5001

# Set the entrypoint command - run app.py from the app directory
CMD ["python", "app/app.py"]