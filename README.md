# mlp — EDA, Pipelines, Reports, Streamlit

Ce projet fournit un squelette modulaire pour auditer un dataset (EDA), comparer des pipelines ML, générer des rapports (Jinja2) et proposer une interface Streamlit multipages avec routage programmatique, en s’appuyant sur Hydra/OmegaConf + Pydantic pour la configuration validée et uv pour un lancement local rapide et reproductible [web:150][web:141][web:198][web:229].

## 1) Générer l’arborescence (create_project_from_yaml.py)
- Placer `project_structure.yaml` à la racine, puis exécuter le script de génération pour créer dossiers/fichiers et appliquer les permissions POSIX (dirs 755, .py/.sh 755, fichiers texte 644) [web:160].  
- Commandes possibles:
  - Python direct: `python create_project_from_yaml.py` [web:160].  
  - Avec uv (éphémère): `uv run --with -r requirements.txt python create_project_from_yaml.py` [web:128].  
  - Avec uv (venv persistant):
    - `uv venv`  
    - `uv pip install -r requirements.txt`  
    - `uv run python create_project_from_yaml.py` [web:128].  
- Format du YAML:
  - Racine `project:` avec deux sections: `directories:` (nœuds pouvant contenir `children:`) et `files:` (fichiers au niveau racine) [web:162].  
  - Section optionnelle `permissions:` pour overrides par motifs (glob) avec modes en octal (ex. `755` ou `"0o755"`) pour fichiers et répertoires [web:160].  

Paramètres attendus (extraits de project_structure.yaml):
- `project.name`: nom logique du projet (utilisé comme dossier sous outputs pour les artefacts) [web:150].  
- `directories.conf`: dossiers de configuration Hydra, avec YAML composables (ex. orchestrateurs) [web:150].  
- `directories.src`: code Python par phases: config, datanalysis (EDA), modeling (pipelines), datavisualization (Jinja), orchestrators, templates [web:148].  
- `directories.streamlit_pages`: pages de l’app Streamlit pour le routage programmatique [web:233].  
- `files.*`: fichiers à créer à la racine (ex. `streamlit_app.py`, `main.py`, `requirements.txt`) [web:229].  
- `permissions.*`: mappage de patterns vers modes (octal), ex. `"*.py": 755`, `"conf/**/*.yaml": 644`, etc. [web:160].  

Bonnes pratiques:
- Versionner `project_structure.yaml` et le script pour reproduire la skeleton build (CI/CD) [web:160].  
- Utiliser uv pour installations rapides et isolées, particulièrement utile en développement et CI [web:128].  

## 2) Projet mlp (architecture et composants)
- Configuration: Hydra/OmegaConf + Pydantic; YAML composables sous `conf/`, validés en schémas Pydantic (`src/config/schemas.py`) via un `ConfigManager` (`src/config/config_manager.py`) [web:150][web:141].  
- EDA: YData Profiling génère un profil HTML; une synthèse JSON et des flags (scaling, n<<p, déséquilibre, colinéarité) complètent l’audit (`src/datanalysis/eda_profile.py`, `src/datanalysis/eda_summary.py`) [web:219][web:108].  
- Pipelines: `PipelineFactory` assemble des pipelines scikit-learn (prétraitements, sélecteurs, PCA/UMAP, estimateurs SVC/RF, AutoML optionnel), `PipelineEvaluator` réalise GridSearchCV/StratifiedKFold et exporte `cv_results_.csv` [web:96][web:133].  
- Reporting: Jinja2 pour séparer rendu/logiciel; `src/templates/` contient les templates, `ReportRenderer` rend HTML/MD, `ReportOrchestrator` orchestre la production [web:198][web:201].  
- Orchestration: `GeneralOrchestrator` exécute EDA → Pipelines → Report et dépose les artefacts dans `outputs/<project_name>/...` [web:150].  

Organisation principale:
- `conf/`: configs Hydra (root + orchestrators/*.yaml) [web:150].  
- `src/config/`: schémas Pydantic + gestionnaire de configuration [web:141].  
- `src/datanalysis/`: génération profil YData + synthèse JSON [web:219].  
- `src/modeling/`: construction/évaluation des pipelines ML [web:96].  
- `src/datavisualization/`: rendu Jinja2 (HTML/MD) [web:198].  
- `src/orchestrators/`: EDA, Pipelines, Report, General [web:150].  
- `src/templates/`: templates Jinja2 (base/report HTML & MD) [web:198].  
- `main.py`: entrée Hydra pour lancer l’orchestration de démonstration [web:150].  

Installation/Exécution (uv recommandé):
- Éphémère: `uv run --with -r requirements.txt python main.py` [web:128].  
- Persistant:
  - `uv venv`  
  - `uv pip install -r requirements.txt`  
  - `uv run python main.py` [web:128].  

Artefacts attendus après run:
- `outputs/<project_name>/eda/`: `eda_summary_*.json`, `profile_*.html` [web:219].  
- `outputs/<project_name>/pipelines/`: `cv_*.csv` [web:96].  
- `outputs/<project_name>/reports/`: `report_*.html`, `report_*.md` [web:198].  

## 3) Interface Streamlit multipages (routage programmatique)
- Entrée: `streamlit_app.py` définit les pages via `st.Page` et `st.navigation`, avec un menu dynamique en code [web:233].  
- Pages: sous `streamlit_pages/` — `home.py`, `eda.py`, `pipelines.py`, `reports.py`; chaque module expose `run()` et utilise `st.cache_data`/`st.cache_resource` pour de bonnes perfs [web:229][web:240].  
- EDA + YData: la page EDA affiche la synthèse JSON et intègre le profil YData via iframe (HTML existant), et propose en option une génération ad hoc (to_html en mémoire) pour démonstration [web:219].  

Lancement avec uv:
- Éphémère:  
  - `uv run --with streamlit --with -r requirements.txt streamlit run streamlit_app.py` [web:229].  
- Persistant:  
  - `uv venv`  
  - `uv pip install -r requirements.txt`  
  - `uv run streamlit run streamlit_app.py` [web:229].  

Conseils:
- Après exécution d’un orchestrateur depuis l’UI (si intégré), vider le cache: `st.cache_data.clear()` et `st.cache_resource.clear()` pour rafraîchir les artefacts affichés [web:240].  
- En déploiement distant, préférer un lien de téléchargement aux iframes `file://` ou servir les HTML YData via URL publique/serveur statique [web:234][web:236].  

Références
- Hydra/OmegaConf: composition YAML, overrides CLI [web:150].  
- Pydantic: validation forte des schémas de configuration [web:141].  
- Jinja2: templates, séparation logique/présentation [web:198][web:201].  
- Streamlit: multipage, st.Page/st.navigation, API de widgets et caches [web:229][web:233][web:240].  
- YData Profiling: génération de profils EDA HTML [web:219][web:108].
