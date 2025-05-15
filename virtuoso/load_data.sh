#!/bin/bash
singularity exec \
  --bind "$(pwd)":/database \
  --bind "$(pwd)/settings":/settings \
  sif/virtuoso_latest.sif \
  isql 1111 dba dba <<'EOF'

ld_dir('/database', 'chunk_*.gz', 'http://www.wikidata.org');

rdf_loader_run();
rdf_loader_run();
rdf_loader_run();
rdf_loader_run();
rdf_loader_run();
rdf_loader_run();
rdf_loader_run();
rdf_loader_run(); 

checkpoint;

EXIT;
EOF
echo "✅ Load commands submitted to Virtuoso. Check DB.DBA.load_list for progress."
