#!/usr/bin/env sh
# Compile tous les .po en .mo avec vérifications et statistiques (compatible /bin/sh).

# Pas de 'pipefail' en POSIX sh
set -eu

ROOT_DIR="${1:-i18n/locales}"

# Motif à vérifier dans les msgid (laisser vide pour désactiver)
# dev -> a retirer en prod (contrôle supplémentaire de contenu)
CHECK_PATTERN="${CHECK_PATTERN:-NAV_}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Erreur: $1 est requis mais introuvable dans PATH." >&2
    exit 2
  }
}

need msgfmt
need msgunfmt
need grep
need find

[ -d "$ROOT_DIR" ] || {
  echo "Erreur: dossier introuvable: $ROOT_DIR" >&2
  exit 1
}

total=0
locales_comptees=0

# Parcourt chaque répertoire LC_MESSAGES
# shellcheck disable=SC2046
for lcd in $(find "$ROOT_DIR" -type d -name "LC_MESSAGES" -print); do
  shcount=0

  # Compile tous les .po du répertoire
  for po in "$lcd"/*.po; do
    [ -e "$po" ] || continue
    mo="${po%.po}.mo"

    echo "Compil.: $po -> $mo"

    # dev -> a retirer en prod (diagnostic: vérifier que le .po contient le motif)
    has_pattern=0
    if [ -n "$CHECK_PATTERN" ] && grep -q "msgid \"$CHECK_PATTERN" "$po"; then
      has_pattern=1
      echo "Vérif .po: motif '$CHECK_PATTERN' détecté dans $(basename "$po")"
    fi

    # Compilation stricte + statistiques (msgfmt écrit les stats sur stderr)
    # POSIX sh: capturer stderr via 2>&1 et taire stdout
    stats="$(msgfmt --check-format --statistics -o "$mo" "$po" 2>&1 >/dev/null)" || {
      echo "ERREUR: échec msgfmt pour $po" >&2
      exit 3
    }
    echo "Stats: $stats"

    # Contrôles post-compilation
    if [ ! -s "$mo" ]; then
      echo "ERREUR: fichier .mo vide ou manquant: $mo" >&2
      exit 4
    fi

    # dev -> a retirer en prod (diagnostic: motif présent dans le .mo si présent dans le .po)
    if [ "$has_pattern" -eq 1 ]; then
      if msgunfmt "$mo" | grep -q "msgid \"$CHECK_PATTERN"; then
        echo "Vérif .mo: motif '$CHECK_PATTERN' trouvé dans $(basename "$mo")"
      else
        echo "ALERTE: motif '$CHECK_PATTERN' présent dans $po mais introuvable dans $mo" >&2
      fi
    fi

    shcount=$((shcount + 1))
    total=$((total + 1))
  done

  if [ "$shcount" -gt 0 ]; then
    locales_comptees=$((locales_comptees + 1))
    echo "Locale $(dirname "$lcd") : $shcount fichier(s) compilé(s)."
  fi
done

echo "Terminé: $total fichier(s) .mo généré(s) sous $ROOT_DIR (${locales_comptees} locale(s))."
