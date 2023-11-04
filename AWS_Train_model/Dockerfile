FROM python:3.10.0-slim
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app/
COPY . /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]