FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /ai_project

# Install system dependencies required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first to leverage Docker layer caching
# Ensure you are using the CPU version of the requirements for Docker!
COPY requirements.txt .

# Install dependencies and clear pip cache to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the application files
COPY . .

# Expose the application port
EXPOSE 5000

# Bind Flask to 0.0.0.0 so the container is accessible from the host machine
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]