#!/bin/sh
# Starts the Cat or Alligator demo on macOS / Linux (works offline).
cd "$(dirname "$0")" && exec python3 tools/serve.py 8000
