#!/bin/bash
singularity exec \
  --bind "$(pwd)":/database \
  --bind "$(pwd)/settings":/settings \
  sif/virtuoso_latest.sif \
  isql 1111 dba dba