# Use a slim Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to speed up builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (src, models, data, etc.)
COPY . .

# Expose the port Flask uses
EXPOSE 5000

# Start the app (Notice we point to src/app.py)
CMD ["python", "src/app.py"]