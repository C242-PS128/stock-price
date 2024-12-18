FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "api.py"]