FROM python:3.12-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY package_aura package_aura
COPY models models

CMD uvicorn package_aura.api_file:app --host 0.0.0.0 --port ${PORT}
