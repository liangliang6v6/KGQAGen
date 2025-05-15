#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wikidata

# INDEX_FILE_DIR=Wikidata/kg/data
# HOST_IP=$2
# BASE_PORT=23546

# > server_urls_new.txt

# for CHUNK in {0..9}; do
#   PORT=$((BASE_PORT + CHUNK))
#   echo "Launching chunk $CHUNK on port $PORT..."

#   python3 Wikidata/simple_wikidata_db/db_deploy/server.py \
#     --data_dir "$INDEX_FILE_DIR" \
#     --chunk_number "$CHUNK" \
#     --host_ip "$HOST_IP" \
#     --port "$PORT" \
#     >> "logs/server_$CHUNK.log" 2>&1 &

#   echo "http://$HOST_IP:$PORT" >> server_urls_new.txt
# done


# python3 Wikidata/simple_wikidata_db/preprocess_dump.py \
#   --input_file Wikidata/latest-all.json.gz \
#   --out_dir Wikidata/kg/data \
#   --batch_size 10000 \
#   --language_id en \

# echo "[*] Building index chunks in parallel..."

# python3 Wikidata/simple_wikidata_db/db_deploy/build_index.py \
#   --input_dir Wikidata/kg/data \
#   --output_dir Wikidata/kg/data/indices \
#   --num_chunks 10 \
#   --num_workers 40

# wait
# echo "[✔] All index chunks built."

# HOST_IP="127.0.0.1"  # or use `hostname -I | awk '{print $1}'` to get local IP

# mkdir -p logs

#!/bin/bash

# mkdir -p logs

python3 Wikidata/simple_wikidata_db/db_deploy/server.py \
  --data_dir Wikidata/kg/data \
  --chunk_number 0 \
  --host_ip 127.0.0.1 \
  --port 23546


# rm server_urls.txt

# for i in {0..9}; do
#     python -u Wikidata/simple_wikidata_db/db_deploy/server.py --data_dir Wikidata/kg/data --chunk_number $i --port 2315$i > logs/server_log_$i.log 2>&1 &
# done

# wait

# python3 Wikidata/simple_wikidata_db/db_deploy/client.py \
#   --addr_list server_urls.txt

