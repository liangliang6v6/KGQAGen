#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Read whatever the old one-time PW was (if it exists)
OLDPW="$( tr -d '\r\n' < settings/dba_password 2>/dev/null || echo )"

# Invoke virtuoso-t directly to reset the password on the existing DB files
singularity exec \
  --bind "$(pwd)":/database \
  sif/virtuoso_latest.sif \
  virtuoso-t \
    +pwdold "${OLDPW}" \
    +pwddba "dba" \
    -c /database/virtuoso.ini

echo "🔑 DBA password has been reset to 'dba'"
