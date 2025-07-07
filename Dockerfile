# 1. Base image
FROM python:3.12-slim

# 2. Set working dir
WORKDIR /app

# 3. Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code (including subsidy.csv!)
COPY . .

# 5. Expose the port Fly will use
EXPOSE 8000

# 6. Start Uvicorn, reading $PORT (set by Fly)
#    Shell form expands $PORT correctly
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
