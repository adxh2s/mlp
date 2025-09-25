FROM python:3.11-slim

# Base Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Outils système (curl, build, gettext pour .po → .mo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential gettext && \
    rm -rf /var/lib/apt/lists/*

# Installer uv (URL brute, ne modifie pas les profils shell)
ENV INSTALLER_NO_MODIFY_PATH=1
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Venv hors bind-mount et PATH explicite
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# (Option de debug à retirer si inutile)
# RUN /bin/sh -lc 'echo "PATH=${PATH}" && which python || true && python -V || true'

# Dépendances de base (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv venv "$VIRTUAL_ENV" && uv sync --frozen --no-dev

# Code + installation du paquet + extras dev
COPY . .
RUN uv pip install -e . && uv pip install -e ".[dev]"

# Compiler les .po en .mo si présents
RUN if [ -d "i18n/locales" ]; then \
      find i18n/locales -name "*.po" -type f -print0 | \
      xargs -0 -I '{}' sh -c 'msgfmt "$1" -o "${1%.po}.mo"' sh '{}'; \
    fi

# Paramètres par défaut consommés par l’UI
ENV MLP_OUTPUTS_DIR=outputs \
    MLP_PROJECT_NAME=demo_project \
    MLP_NOTEBOOKS_DIR=notebooks \
    MLP_NOTEBOOKS_URL= \
    MLP_LANG=fr

# UI Streamlit
EXPOSE 8501

# Lancement de l'app (uv run utilise le venv)
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
