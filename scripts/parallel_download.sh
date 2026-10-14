#!/usr/bin/env bash
# parallel_download.sh URL OUT [PARTS] [MD5]: split a ranged HTTP download into PARTS
# concurrent curl requests (resumable per part), concatenate, verify optional MD5.
set -eu
URL=$1; OUT=$2; PARTS=${3:-8}; MD5=${4:-}
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
pids=()
for i in $(seq 0 $((PARTS - 1))); do
  start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 )); [ $end -ge $SIZE ] && end=$((SIZE - 1))
  part="$OUT.part$i"; have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
  want=$(( end - start + 1 ))
  if [ "$have" -lt "$want" ]; then
    curl -sSfL --retry 5 -r "$((start + have))-$end" "$URL" >> "$part" & pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
parts=(); for i in $(seq 0 $((PARTS - 1))); do parts+=("$OUT.part$i"); done
cat "${parts[@]}" > "$OUT.tmp"
[ "$(stat -c %s "$OUT.tmp")" -eq "$SIZE" ] || { echo "size mismatch"; exit 1; }
if [ -n "$MD5" ]; then echo "$MD5  $OUT.tmp" | md5sum -c - >/dev/null || { echo "MD5 mismatch"; exit 1; }; fi
mv "$OUT.tmp" "$OUT"; rm -f "$OUT".part*
echo "DOWNLOADED $OUT ($SIZE bytes)"
