# Use the official Python image as a base
FROM python:3.8-slim

# Install ollama using pip
RUN pip install ollama

# Expose the default port for ollama
EXPOSE 80

# Run ollama and bind to all interfaces to expose the IP address
CMD ["ollama", "--host", "0.0.0.0"]