for p in {23546..23555}; do
  f=$(lsof -tiTCP:$p); [ -n "$f" ] && kill -9 $f;
done