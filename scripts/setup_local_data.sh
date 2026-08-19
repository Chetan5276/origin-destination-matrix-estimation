#!/usr/bin/env bash
# Ensure data/ links to network, base OD, and the 100k synthetic tensor when present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p data

link_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    ln -sfn "$(realpath "$src")" "$dest"
    echo "linked: $dest -> $src"
  else
    echo "missing source: $src"
  fi
}

link_if_exists "$REPO_ROOT/sioux-falls.net.xml" "data/sioux-falls.net.xml"
link_if_exists "$REPO_ROOT/EstimatedODMatrix.npy" "data/EstimatedODMatrix.npy"
link_if_exists "$REPO_ROOT/outputs/od_generator_100k/synthetic_od_100k.npy" "data/synthetic_od_100000.npy"

echo "Done."
