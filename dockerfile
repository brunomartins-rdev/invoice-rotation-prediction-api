FROM python:3.9-slim

# Set wd
WORKDIR /app

# Copy reqs
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Port
EXPOSE 5000

# Run the API
CMD ["python", "api.py"]
