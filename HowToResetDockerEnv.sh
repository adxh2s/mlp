# Arrêter et supprimer services/ressources du projet: 
docker compose down --volumes --remove-orphans 
# pour s’assurer qu’aucun ancien conteneur n’est réutilisé au prochain up.

# Purger images intermédiaires et caches: 
docker system prune -af 
# pour forcer un rebuild total des couches lors du prochain build.

# Rebuild sans cache et avec pull: 
docker compose build --no-cache --pull app notebooks voila 
# afin de reconstruire toutes les images de services avec le Dockerfile corrigé (ENV PATH et VIRTUAL_ENV).

# Relancer en recréant les conteneurs: 
docker compose up -d --force-recreate app notebooks voila 
# pour être certain que les nouveaux envs sont pris en compte.

# Inspecter l’environnement effectif: 
docker compose exec app sh -lc 'env | grep -E "^(PATH|VIRTUAL_ENV)="' 
# doit afficher VIRTUAL_ENV=/app/.venv et PATH commençant par /app/.venv/bin .

# Vérifier l’interpréteur: 
docker compose exec app sh -lc 'python -c "import sys; print(sys.executable, sys.prefix)"' 
# doit renvoyer /app/.venv/bin/python et /app/.venv si le venv est correctement actif dans l’image.

# Confirmer ruff et outils dev: 
docker compose exec app sh -lc 'which ruff && ruff --version' 
# pour s’assurer que les “console scripts” de l’extra dev sont résolus via /app/.venv/bin.

# Suivre les logs en temps réel: 
docker compose logs -f app notebooks voila 
# permet de lire toutes les sorties de démarrage et d’éventuels messages d’erreur, PATH n’y est pas imprimé par défaut mais les erreurs de résolution de modules y apparaîtront.

# Interroger un service spécifique: 
docker compose logs -f app 
# pour uniquement l’UI Streamlit, et 
docker compose logs -f notebooks 
# pour Jupyter Lab, utile pour confirmer la commande de lancement effective et le bon port.

