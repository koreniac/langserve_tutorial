FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
