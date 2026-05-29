#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Running VULGARIS company showcase..."
python demo/company_showcase.py "$@"
echo ""
echo "Open leave-behind: demo/output/showcase_report.md"
