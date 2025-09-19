FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY conf ./conf
COPY src ./src
COPY streamlit_pages ./streamlit_pages
COPY src/templates ./src/templates

# Install
RUN pip install --upgrade pip && pip install -e .[dev]

EXPOSE 8501

# Default command: Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
