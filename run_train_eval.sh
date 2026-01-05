#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting training..."
python train.py

echo ""
echo "Training complete. Starting gait evaluation..."
python evaluate_gait.py

echo ""
echo "Gait evaluation complete. Starting plot and table generation..."
python evaluate_flat.py

echo ""
echo "Done."
