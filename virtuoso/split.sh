bzip2 -dc latest-truthy.nt.bz2 \
  | split \
      --bytes=20G \
      --numeric-suffixes=1 \
      --suffix-length=2 \
      --filter='gzip -9 > $FILE.gz' \
      - chunk_