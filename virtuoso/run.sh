#!/bin/bash
# singularity pull virtuoso_latest.sif docker://openlink/virtuoso-opensource-7

set -euo pipefail

# Bind two host dirs into the container
singularity exec \
  --bind "$(pwd)":/database \
  --bind "$(pwd)/settings":/settings \
  sif/virtuoso_latest.sif \
  /virtuoso-entrypoint.sh \
  # virtuoso-t -c /database/virtuoso.ini --foreground