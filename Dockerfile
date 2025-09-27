FROM python:3.11-slim

# Base Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Outils système (curl, build, gettext pour .po → .mo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential gettext && \
    rm -rf /var/lib/apt/lists/*

# Installer uv (sans modifier le PATH automatiquement)
RUN curl -fsSL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh

# Venv géré par uv et PATH explicite
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Forcer uv à cibler /opt/venv
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Dépendances verrouillées (dev inclus en dev)
COPY pyproject.toml uv.lock ./
RUN /root/.local/bin/uv venv "/opt/venv" && \
    /root/.local/bin/uv sync --frozen --all-extras --dev --python /opt/venv/bin/python  # dev -> a retirer en prod (retirer --dev)

# Code & assets: docs + i18n (présents dès le build)
COPY docs ./docs
COPY i18n ./i18n
COPY . .
RUN /root/.local/bin/uv pip install --no-deps -e .

# Dupliquer docs dans static pour image autonome (prod); neutralise symlink existant
RUN rm -rf /app/static/docs && mkdir -p /app/static/docs && cp -a /app/docs/. /app/static/docs/

# Compiler les .po en .mo au build (pas de recompilation au lancement)
RUN if [ -d "i18n/locales" ]; then \
      find i18n/locales -name "*.po" -type f -print0 | \
      xargs -0 -I '{}' sh -c 'msgfmt "$1" -o "${1%.po}.mo"' sh '{}'; \
    fi

# Paramètres par défaut (inclut la static serving)
ENV MLP_OUTPUTS_DIR=outputs \
    MLP_PROJECT_NAME=demo_project \
    MLP_NOTEBOOKS_DIR=notebooks \
    MLP_NOTEBOOKS_URL= \
    MLP_LANG=fr \
    MLP_DOCS_DIR=docs \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

EXPOSE 8501

# Lancer Streamlit via /opt/venv
CMD ["/opt/venv/bin/python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
