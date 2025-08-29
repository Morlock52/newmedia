#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

files=$(rg -l "console\.(log|warn|error|info|debug)\(" api --hidden || true)

for file in $files; do
  echo "Patching $file"
  # insert logger require if not present
  if ! rg -n "\blogger\b" "$file" >/dev/null 2>&1; then
    rel=$(python3 - <<PY
import os,sys
file=sys.argv[1]
rel=os.path.relpath(os.path.join(os.getcwd(),'middleware','logger.js'), start=os.path.dirname(file))
print(rel.replace('\\\\','/'))
PY
 "$file")
    if [[ $(head -n1 "$file") == \#\!* ]]; then
      # keep shebang
      (head -n1 "$file"; echo "const logger = require('$rel'.startsWith('.') ? '$rel' : './$rel');"; tail -n +2 "$file") > "$file.tmp"
    else
      (echo "const logger = require('$rel'.startsWith('.') ? '$rel' : './$rel');"; cat "$file") > "$file.tmp"
    fi
    mv "$file.tmp" "$file"
  fi

  # replace console.* with logger.*
  sed -E -i '' \
    -e "s/console\\.log\s*\(/logger.info(/g" \
    -e "s/console\\.info\s*\(/logger.info(/g" \
    -e "s/console\\.warn\s*\(/logger.warn(/g" \
    -e "s/console\\.error\s*\(/logger.error(/g" \
    -e "s/console\\.debug\s*\(/logger.debug(/g" \
    "$file" || true

done
