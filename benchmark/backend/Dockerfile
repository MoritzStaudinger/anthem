FROM python:3.11-slim

WORKDIR /app

COPY ./backend /app/backend
COPY ./app.py /app/
COPY ./backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "300", "app:app"]
