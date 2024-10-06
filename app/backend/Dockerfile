# Use the official Python image as the base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
