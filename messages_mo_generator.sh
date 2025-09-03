#!/usr/bin/env sh
# compile_mo_all.sh - Compile tous les .po en .mo pour chaque langue/domaine.

set -eu

ROOT_DIR="i18n/locales"

if [ ! -d "$ROOT_DIR" ]; then
  echo "Erreur: dossier introuvable: $ROOT_DIR" >&2
  exit 1
fi

if ! command -v msgfmt >/dev/null 2>&1; then
  echo "Erreur: msgfmt (gettext) est requis mais introuvable dans PATH." >&2
  exit 2
fi

total=0
# Recurse sur chaque locale
find "$ROOT_DIR" -type d -name "LC_MESSAGES" | while IFS= read -r lcd; do
  shcount=0
  for po in "$lcd"/*.po; do
    [ -e "$po" ] || continue
    mo="${po%.po}.mo"
    echo "Compil.: $po -> $mo"
    msgfmt -o "$mo" "$po"
    shcount=$((shcount + 1))
    total=$((total + 1))
  done
  if [ "$shcount" -gt 0 ]; then
    echo "Locale $(dirname "$lcd") : $shcount fichier(s) compilé(s)."
  fi
done

echo "Terminé: $total fichier(s) .mo généré(s) sous $ROOT_DIR."
