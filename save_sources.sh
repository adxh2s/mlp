#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"  # [3]
INCLUDE_DIR="src"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="code_sources_${TS}.tar.gz"
ARCHIVE_PATH="${ROOT_DIR}/${ARCHIVE_NAME}"

cd "${ROOT_DIR}"

# Dossier de staging à plat
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

# Fonction de copie avec gestion des collisions
copy_flat() {
  local src="$1"
  local base
  base="$(basename "$src")"
  local dest="${STAGE_DIR}/${base}"
  if [[ -e "$dest" ]]; then
    # Renommer en ajoutant un suffixe unique
    local name="${base%.*}"
    local ext="${base##*.}"
    if [[ "$name" == "$ext" ]]; then ext=""; else ext=".$ext"; fi
    local i=1
    while [[ -e "${STAGE_DIR}/${name}-${i}${ext}" ]]; do ((i++)); done
    dest="${STAGE_DIR}/${name}-${i}${ext}"
  fi
  cp -p "$src" "$dest"
}

# 1) Fichiers .py/.yaml/.yml à la racine (non récursif)
for f in ./*.py ./*.yaml ./*.yml; do
  [[ -f "$f" ]] && copy_flat "$f"
done

# 2) Fichiers .py/.yaml/.yml sous src (récursif)
if [[ -d "$INCLUDE_DIR" ]]; then
  while IFS= read -r -d '' f; do
    copy_flat "$f"
  done < <(find "$INCLUDE_DIR" -type f \( -name '*.py' -o -iname '*.yaml' -o -iname '*.yml' \) -print0)
fi

# Créer archive plate: tar du staging avec -C et .
tar -C "$STAGE_DIR" -czf "$ARCHIVE_PATH" .  # [4]

echo "Archive créée: $ARCHIVE_PATH"
