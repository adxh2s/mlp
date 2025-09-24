# mlp — EDA, Pipelines, Reports, Streamlit

Squelette modulaire pour auditer un dataset (EDA), comparer des pipelines ML, générer des rapports (Jinja2) et exposer une interface Streamlit multipages, avec configuration Hydra/OmegaConf validée par Pydantic et instrumentation de logs centralisée.  

## Aperçu

- EDA: génération d’un profil HTML et d’un récapitulatif JSON avec flags d’aide à la modélisation.  
- Pipelines: assemblage/évaluation scikit-learn (prétraitement, sélecteurs, réduction, estimateurs) avec export des résultats CV.  
- Reporting: rendu HTML/Markdown depuis templates Jinja2.  
- Orchestration: enchaînement EDA → Pipelines → Report, artefacts sous outputs/<project>.  
- UI: application Streamlit multipages avec routage programmatique et I18n minimal.  

## Arborescence (principaux dossiers)

- conf/: YAML Hydra (racine + orchestrators/config|data|eda|file|messages|pipelines|report).  
- src/: code en sous‑packages (config, datanalysis, modeling, datavisualization, evaluation, preprocessing, instrumentation, orchestrators, templates, ui).  
- streamlit_pages/: pages Streamlit (home, eda, pipelines, reports, utils_runs).  
- data/: dossiers d’entrée/sortie données.  
- outputs/: artefacts générés (EDA, pipelines, rapports).  

## Installation

### Option uv (recommandé)
- Éphémère:  
  uv run --with -r requirements.txt python main.py  
- Persistant:  
  uv venv  
  uv pip install -r requirements.txt  
  uv run python main.py  

## Configuration (Hydra + Pydantic)

YAML de base: conf/config.yaml avec defaults, project, logger, mlflow.  
Orchestrateurs: conf/orchestrators/<domain>/<file>.yaml (config, data, eda, file, messages, pipelines, report).  
Validation: src/config/schemas.py et gestion via ConfigManager (instrumentation).  

## Orchestration en CLI

- Point d’entrée: main.py  
- Exemples:  
  python main.py  
  python main.py orchestrators.pipelines.active=baseline  
  python main.py data.sep=";" file.save_input_file=true  

Les artefacts sont déposés dans outputs/<project>/eda|pipelines|reports selon les étapes exécutées.  

## UI Streamlit

- Entrée: streamlit_app.py  
- Démarrage (uv):  
  uv run streamlit run streamlit_app.py  
- Pages: streamlit_pages/{home, eda, pipelines, reports}.  
- I18n basique et messages via instrumentation/messages.  

## Développement

- Qualité: Ruff, MyPy, PyTest, Pre-commit (configurés dans pyproject.toml).  
- Tests: tests/* (unitaires et intégration).  
- Génération du contexte: python session_context_md.py → session-context.md.  

## Composants clés (extraits)

- Instrumentation: bootstrap_logging, logger_manager (+ structlog), logger_factory, config_manager, file_manager, messages_manager, logger_mixin.  
- EDA: datanalysis/eda_profile.py, datanalysis/eda_summary.py.  
- Pipelines: modeling/pipelines/{factory,evaluator,consts}.py.  
- Deep Learning (optionnel): modeling/dl/{config,consts,factory,trainer}.py.  
- Reporting: datavisualization/report_renderer.py + src/templates/*.jinja.  
- Orchestrateurs: orchestrators/{config,data,eda,file,messages,pipelines,report,general,app}.py.  

## Artefacts

- EDA: outputs/<project>/eda/{profile_*.html, eda_summary_*.json}.  
- Pipelines: outputs/<project>/pipelines/cv_*.csv.  
- Rapports: outputs/<project>/reports/{report_*.html, report_*.md}.  
