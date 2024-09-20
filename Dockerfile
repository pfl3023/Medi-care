# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app
CMD ["uvicorn", "RAG.retriever:app", "--host", "0.0.0.0", "--port", "5000"]
