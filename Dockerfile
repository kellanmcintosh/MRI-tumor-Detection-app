FROM python:3.11-slim

WORKDIR /code

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY scripts/ scripts/
COPY static/ static/

# Pulls the pinned HF Hub model revision into app/model/ at build time, so
# the running container never needs network access to HF Hub.
RUN PYTHONPATH=/code python scripts/download_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
