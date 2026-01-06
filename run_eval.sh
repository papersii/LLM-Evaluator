#!/bin/bash

# LLM Evaluator - Generic Execution Script
# Usage: ./run_eval.sh [data_file]
# Example: ./run_eval.sh data/test_cases.jsonl

set -e  # Exit on error

# Determine Python executable (prefer virtual environment)
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found. Please install Python or activate your virtual environment."
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"
echo ""

# Set data path (default or from argument)
DATA_PATH="${1:-data/test_cases.jsonl}"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file '$DATA_PATH' not found."
    exit 1
fi

# Run evaluation
echo "Starting evaluation with data: $DATA_PATH"
echo "----------------------------------------"
$PYTHON main.py --data_path "$DATA_PATH"