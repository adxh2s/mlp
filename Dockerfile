FROM python:3.11-slim

# Base Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Créer un utilisateur non-root paramétrable
ARG UID=1001
ARG GID=1001
ARG USER=adxh2s
RUN groupadd -g ${GID} ${USER} && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

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

# === Logs centralisés ===
# - Créer /logs avec permissions larges (sur-monté en volume en dev/prod)
# - Exposer MLP_LOG_FILE pour le LoggerOrchestrator/LoggerManager
RUN mkdir -p /logs && chmod -R 777 /logs
ENV MLP_LOG_FILE=/logs/streamlit_app.log
VOLUME ["/logs"]

# Dépendances verrouillées
COPY pyproject.toml uv.lock ./
RUN /root/.local/bin/uv venv "/opt/venv" && \
    /root/.local/bin/uv sync --frozen --all-extras --python /opt/venv/bin/python

# Copier tout le code (inclut mlp/i18n et docs/)
COPY . .

# Installer le paquet applicatif sans réinstaller les deps
RUN /root/.local/bin/uv pip install --no-deps -e .

# i18n: compiler les .po → .mo à partir de mlp/i18n/locales et
# dupliquer sous /app/i18n pour compatibilité runtime (localedir="i18n/locales")
RUN set -eux; \
    if [ -d "mlp/i18n/locales" ]; then \
      # Garde: s'assurer que NAV_ existe dans le .po (évite d'embarquer un catalogue obsolète)
      grep -Rqn --include="*.po" 'msgid "NAV_' mlp/i18n/locales || { echo "NAV_ absents dans les .po"; exit 1; }; \
      # Compilation stricte + statistiques
      find mlp/i18n/locales -type f -name '*.po' -print0 | \
      xargs -0 -I '{}' sh -c 'msgfmt --check-format --statistics "$1" -o "${1%.po}.mo"' sh '{}'; \
      # Recopie vers /app/i18n pour aligner le chemin de recherche runtime
      rm -rf /app/i18n && mkdir -p /app/i18n && cp -a mlp/i18n/. /app/i18n/; \
    fi

# Static: image autonome (prod) — servir docs/ via app/static/docs
RUN rm -rf /app/static/docs && mkdir -p /app/static/docs && \
    if [ -d "/app/docs" ]; then cp -a /app/docs/. /app/static/docs/; fi

# Paramètres par défaut (inclut la static serving de Streamlit)
ENV MLP_OUTPUTS_DIR=outputs \
    MLP_PROJECT_NAME=demo_project \
    MLP_NOTEBOOKS_DIR=notebooks \
    MLP_NOTEBOOKS_URL= \
    MLP_LANG=fr \
    MLP_DOCS_DIR=docs \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

# Donner la propriété du projet à l'utilisateur non-root (si COPY n'a pas suffi)
RUN chown -R ${USER}:${USER} /app

USER ${USER}

EXPOSE 8501

# Lancer Streamlit via /opt/venv
CMD ["/opt/venv/bin/python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
