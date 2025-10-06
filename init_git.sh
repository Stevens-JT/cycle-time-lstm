#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".git" ]; then
  git init
fi
git add .
git commit -m "Initial commit: cycle-time project"
echo "Git repo initialized and initial commit created."
