#!/bin/sh
# Forced command for the "dimension42-L6-transfer" SSH keys (authorized_keys on falke64).
# The only thing these keys can do: store one compressed result chunk from stdin under
# ~/dimension42-explorer/L6/xx/yy.zst. Any other request is refused.
p="$SSH_ORIGINAL_COMMAND"
case "$p" in
  L6/[0-9a-f][0-9a-f]/[0-9a-f][0-9a-f].zst) ;;
  *) echo "refused: $p" >&2; exit 1 ;;
esac
cd /home/ender/dimension42-explorer || exit 1
mkdir -p "${p%/*}" && cat > "$p.part" && mv "$p.part" "$p"
