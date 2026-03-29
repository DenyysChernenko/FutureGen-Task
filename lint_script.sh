#!/bin/bash

echo "Starting code quality checks..."
echo ""

# Run Black
echo "Running Black..."
black src/ scripts/ tests/

# Run Ruff
echo "Running Ruff..."
ruff check src/ scripts/ tests/ --fix

# Run MyPy
echo "Running MyPy..."
mypy src/ tests/ --ignore-missing-imports

echo ""
echo "All checks completed!"
