FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt pyproject.toml ./
COPY server/ server/
COPY __init__.py ./
RUN pip install --no-cache-dir .

# Copy remaining files
COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]