#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wikidata
DATA_DIR="Wikidata/kg/data"
BASE_PORT=23546
SERVER_SCRIPT="Wikidata/simple_wikidata_db/db_deploy/server.py"
LOG_DIR="logs"
ADDR_FILE="server_urls.txt"

for i in {1..10}; do ls -lh $DATA_DIR/indices/*_chunk_${i}.pickle; done


mkdir -p $LOG_DIR
rm -f $ADDR_FILE

for CHUNK in {1..3}
do
  PORT=$((BASE_PORT + CHUNK))
  echo "Launching chunk $CHUNK on port $PORT..."
  python $SERVER_SCRIPT \
    --data_dir $DATA_DIR \
    --chunk_number $CHUNK \
    --port $PORT \
    --host_ip 127.0.0.1 \
    > $LOG_DIR/server_$CHUNK.log 2>&1 &
  echo "http://127.0.0.1:$PORT" >> $ADDR_FILE
done
