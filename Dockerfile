FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Outils système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential gettext && \
    rm -rf /var/lib/apt/lists/*

# Installer uv (ne touche pas les profils) puis l’exposer dans PATH
RUN curl -fsSL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Vérification (optionnel)
RUN /root/.local/bin/uv --version

# Dépendances via lock (incluant dev + extras)
COPY pyproject.toml uv.lock ./
RUN /root/.local/bin/uv venv "$VIRTUAL_ENV" && \
    /root/.local/bin/uv sync --frozen --all-extras --dev

# Code + docs, installation du paquet en editable sans deps
COPY docs ./docs
COPY . .
RUN /root/.local/bin/uv pip install --no-deps -e .

# Static file serving: symlink pour exposer docs/ sous app/static/docs
RUN mkdir -p /app/static && ln -s /app/docs /app/static/docs

# Compiler les .po en .mo si présents
RUN if [ -d "i18n/locales" ]; then \
      find i18n/locales -name "*.po" -type f -print0 | \
      xargs -0 -I '{}' sh -c 'msgfmt "$1" -o "${1%.po}.mo"' sh '{}'; \
    fi

# Paramètres par défaut Streamlit
ENV MLP_OUTPUTS_DIR=outputs \
    MLP_PROJECT_NAME=demo_project \
    MLP_NOTEBOOKS_DIR=notebooks \
    MLP_NOTEBOOKS_URL= \
    MLP_LANG=fr \
    MLP_DOCS_DIR=docs \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
