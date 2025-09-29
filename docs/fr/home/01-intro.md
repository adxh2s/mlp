# NBA — Exploration et pré‑processing des tirs

Analyse exploratoire, data visualisation et préparation des données sur les tirs de joueurs NBA (4 dernières saisons), avec création de variables contextuelles et premières modélisations [source: Rapport d’exploration].

## Contexte métier
L’analyse des tirs NBA aide à optimiser la performance individuelle et les stratégies collectives en exploitant types de tirs, localisation, tempo et interactions entre joueurs.  
Les données riches issues du suivi temps réel permettent de caractériser styles de jeu, comportements gagnants et décisions terrain.

## Contexte technique
Source: API officielle nba_api (4 dernières saisons), plus actuelle que des jeux statiques (ex. Kaggle).  
Endpoints utilisés:
- playercareerstats — statistiques globales par joueur  
- shotchartdetail — détails précis des tirs  
- PlayByPlayV2 — chronologie d’événements  
- players — informations descriptives des joueurs

Environ 37 000 tirs et >70 variables initiales; important travail de pré‑processing, feature engineering et réduction de dimension.

## Contexte économique
L’anticipation fine des performances influence recrutement, préparation tactique et valorisation joueurs/franchises (avantage compétitif tangible).

## Contexte scientifique
Approche “sports analytics” appliquant le machine learning pour transformer des événements (tir) en variables explicatives pour des modèles prédictifs, en intégrant dynamique temporelle et contexte.

## Objectifs du projet
- Explorer et visualiser 4 saisons de tirs NBA  
- Construire des variables cumulées, temporelles et contextuelles  
- Identifier patterns: types/zones préférées, séquences réussite/échec  
- Préparer un dataset pour modélisation supervisée (probabilité de réussite d’un tir)

## Données — cadre général
- Import via nba_api (officiel), données publiques  
- Couverture: 4 dernières saisons  
- Volume: plusieurs centaines de milliers d’événements (tirs + actions)

### Variables clés (extraits)
| Variable | Description |
|---|---|
| GAME_ID | Identifiant unique du match |
| PLAYER_ID | Identifiant joueur |
| TEAM_ID | Identifiant équipe |
| PERIOD_SHOT | Période de jeu (Q1…Q4/OT) |
| ACTION_TYPE | Type d’action (jump shot, dunk…) |
| SHOT_TYPE | Catégorie (2pts/3pts) |
| SHOT_ZONE_BASIC | Zone générale (raquette, périmètre) |
| SHOT_ZONE_AREA | Zone détaillée |
| SHOT_ZONE_RANGE | Distance (plages) |
| SHOT_DISTANCE | Distance en pieds |
| SEASON / SEASON_TYPE | Saison et type (RS/PO) |
| SCOREMARGIN | Écart au moment du tir |
| PREV_EVENTMSGTYPE / ACTIONTYPE | Événement précédent |
| AGE | Âge du joueur |

## Visualisations et statistiques
- Les variables qualitatives (ACTION_TYPE, SHOT_TYPE, SHOT_ZONE_*) montrent des corrélations cohérentes (styles de jeu distincts visibles sur heatmaps).  
- Les variables quantitatives (SHOT_DISTANCE) s’alignent avec SHOT_ZONE_RANGE et SHOT_TYPE; distributions séparées entre 2pts/3pts et apport de l’angle de tir et de la progression d’événement.

### Lien avec la cible (SHOT_MADE_FLAG)
- Réussite plus élevée en Restricted Area; plus faible au périmètre/au‑delà de 3pts (confirmé par tests de Chi²).  
- Dépendances significatives entre réussite, zones de tir et type d’action.

#### Test Chi² — variables catégorielles (extrait)
| Rang | Variable | Chi² | p‑value |
|---:|---|---:|---:|
| 1 | SHOT_ZONE_BASIC | 3232.03 | 0.000000e+00 |
| 2 | ACTION_TYPE | 2592.02 | 0.000000e+00 |
| 3 | SHOT_ZONE_RANGE | 1959.53 | 0.000000e+00 |
| 4 | SHOT_TYPE | 981.47 | 1.92e-215 |
| 5 | SHOT_ZONE_AREA | 594.61 | 2.49e-131 |

- SHOT_DISTANCE est clé: relation négative modérée avec la réussite (corrélation); plus c’est court, plus la probabilité de “made” est élevée.

## Outliers et distributions
- Forte concentration près du panier et au‑delà de la ligne à 3pts; majorité de jump shots.  
- Quelques distributions spécifiques (volumes faibles par zones) générant des outliers; filtrage/regroupements et normalisation appliqués.

## Données manquantes et cohérence
- Peu de valeurs manquantes à l’origine; des NA peuvent apparaître lors de la création de variables cumulées/rolling (manque d’historique en début de série).  
- Stratégies: imputations à 0 pour cumuls et SCOREMARGIN; pas d’incohérences majeures détectées après contrôles (dataset jugé fiable).

## Pré‑processing et feature engineering
- Variables dérivées/chronologiques:  
  - GAME_NUMBER_PLAYER, DAYS_SINCE_LAST_GAME  
  - EVENT_PROGRESSION  
  - WCTIMESTRING_MINUTES, PCTIMESTRING_SECONDS  
  - ANGLE_TO_HOOP  
  - SCOREMARGIN en float, NA→0
- Suppressions: noms joueurs/équipes, LOC_X/LOC_Y, flags redondants
- Statistiques cumulées/rolling par joueur:  
  - Par type/zone (ACTION_TYPE, SHOT_TYPE, SHOT_ZONE_BASIC/AREA/RANGE): *_CUM_ATTEMPTS, *_CUM_MADE, *_CUM_PCT, *_NO_HISTORY  
  - Globaux: TOTAL_CUM_*  
  - Rolling 5 tirs: LAST_5_*  
  - Rolling 5 jours: LAST_5D_*
- Fusion de ces indicateurs dans le dataset pour fournir l’historique utile à chaque tir.

## Modélisation — Random Forest (v1)
Pré‑traitement via ColumnTransformer:  
- Standardisation numériques (SHOT_DISTANCE, SCOREMARGIN, AGE, cumuls, temporelles, angulaires)  
- Encodage one‑hot catégorielles; passage direct des booléens  
- Entraînement Random Forest

### Résultats (jeu test)
| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| Miss | 0.71 | 0.87 | 0.78 | 3707 |
| Made | 0.83 | 0.65 | 0.73 | 3752 |
Accuracy globale ≈ 0.76 (macro/weighted ≈ 0.76–0.77).

Lecture: bon équilibre précision/rappel; meilleure détection des tirs ratés que des tirs réussis.

## Conclusions et suites
- Données riches, nécessitant nettoyage/transformations et variables contextuelles pour le ML.  
- Intégrer cumul et contexte (fenêtres mobiles) est essentiel pour la dimension temporelle.  
- Pistes d’amélioration: feature engineering/selection, tuning RF, comparaison XGBoost/LogReg/SVM, analyse d’importances pour comprendre les facteurs de réussite.

