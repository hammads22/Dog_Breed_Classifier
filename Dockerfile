FROM python:3.10-slim



# Set the working directory
WORKDIR /ai_project

# Copy the application files
COPY . /ai_project/

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 



# Expose the application port
EXPOSE 5000

# Define the command to run the app
CMD ["python", "app.py"]
