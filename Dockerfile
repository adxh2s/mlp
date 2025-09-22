# Dockerfile
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Eviter les compilations locales inutiles et réduire le bruit
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Mettez vos dépendances ici
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Met à jour pip et installe les dépendances du projet
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install -r /workspace/requirements.txt

# Optionnel: créer un utilisateur non-root pour éviter que les fichiers montés soient root
# ARG UID=1000 GID=1000
# RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app
# USER app

# Commande par défaut interactive
CMD ["/bin/bash"]
