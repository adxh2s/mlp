FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential gettext && \
    rm -rf /var/lib/apt/lists/*

# Installer uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Dépendances (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:${PATH}"

# Code + i18n
COPY . .
RUN uv pip install -e .

# Compiler les .po en .mo (si présents)
RUN find i18n/locales -name "*.po" -type f -print0 | xargs -0 -I {} sh -c 'msgfmt "{}" -o "${0%.po}.mo"' {}

ENV MLP_OUTPUTS_DIR=outputs \
    MLP_PROJECT_NAME=demo_project \
    MLP_NOTEBOOKS_DIR=notebooks \
    MLP_NOTEBOOKS_URL= \
    MLP_LANG=fr

EXPOSE 8501 8888

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
