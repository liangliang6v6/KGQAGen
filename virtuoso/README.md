# 🚀 Deploying Wikidata with Virtuoso

This guide walks you through deploying a Virtuoso SPARQL endpoint using the latest Wikidata truthy dump.

## 🔧 Requirements

- `bzip2`, `gzip`, `split`  
- `singularity` or `apptainer` (container runtime)  
- ~1 TB disk space for full dump  
- ≥32 GB RAM recommended  

---

## 📥 Step 1: Download and Preprocess the Dump

```bash
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2
bzip2 -dk latest-truthy.nt.bz2  # Extracts ~955 GB N-Triples file
```

---

## 📂 Step 2: Split into Gzipped Chunks (20 GB Each)

Use streaming to avoid storing the full `.nt` file at once:

```bash
bzip2 -dc latest-truthy.nt.bz2 \
  | split \
      --bytes=20G \
      --numeric-suffixes=1 \
      --suffix-length=2 \
      --filter='gzip -9 > $FILE.gz' \
      - chunk_
```

This produces:
```
chunk_01.gz
chunk_02.gz
...
```

Place the resulting `chunk_*.gz` files into the `db/` directory.

---

## 📦 Step 3: Set Up Virtuoso with Singularity

### 3.1 Pull the Virtuoso Container

```bash
singularity pull virtuoso_latest.sif docker://openlink/virtuoso-opensource-7
```

### 3.2 Prepare Directory Structure

```bash
mkdir -p db settings sif
mv virtuoso_latest.sif sif/
echo "dba" > settings/dba_password  # Replace with a secure password if desired
```

---

## ▶️ Step 4: Start the Virtuoso Server

run `run.sh`:

```bash
#!/bin/bash
set -euo pipefail

singularity exec \
  --bind "$(pwd)/db":/database \
  --bind "$(pwd)/settings":/settings \
  sif/virtuoso_latest.sif \
  /virtuoso-entrypoint.sh
```


## 📥 Step 5: Load RDF Data into Virtuoso

Load wikidata to Virtuso `load_data.sh`:

```bash
#!/bin/bash
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
```

Check loading status, run `monitor.sh`, the in the `isql`, run:

```sql
SELECT ll_state,COUNT(*) FROM DB.DBA.load_list GROUP BY ll_state;
SELECT ll_file, ll_state FROM DB.DBA.load_list;
```
Status codes for `ll_state`:
- `0`: Pending(no start)
- `1`: Inprogress (current running)
- `2`: Completed.

In the interactive `isql` shell, run:

```sql
ld_dir('/database','chunk_*.gz','http://www.wikidata.org');
rdf_loader_run();   -- Repeat in parallel sessions for speed
checkpoint;
```

---

## 🌐 Accessing the SPARQL Endpoint

Once running, point your browser or client to:

```
http://localhost:8890/sparql
```

Test with:

```bash
curl "http://localhost:8890/sparql?query=SELECT%20*%20WHERE%20%7B%20?s%20?p%20?o%20%7D%20LIMIT%2010"
```

---

## ✅ Tips & Tricks

- **Log Monitoring**: `tail -f db/virtuoso.log`  
- **Checkpoints**: Always run `checkpoint;` after data loads  
- **Parallel Ingestion**: Open multiple `isql` shells and repeat `rdf_loader_run();`  
- **Restart**: Simply rerun `./run.sh` to bring the server back up  

---

## 📚 References

- [Virtuoso Open-Source Edition](https://github.com/openlink/virtuoso-opensource)  
- [Wikidata RDF Dumps](https://dumps.wikimedia.org/wikidatawiki/entities/)  
