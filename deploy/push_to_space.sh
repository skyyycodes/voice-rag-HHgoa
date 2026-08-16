#!/usr/bin/env bash
# Push this repo to a Hugging Face Space (Gradio SDK).
#
# Docker Spaces are a paid feature, so the deployed demo runs on the free Gradio
# SDK. Only the presentation layer differs — `app.py` calls the same Pipeline
# in-process that `vrag.server` serves over HTTP.
#
#   ./deploy/push_to_space.sh <hf-username> <space-name>
#
# The index (~300MB) is committed via git-lfs. That is the whole point of
# building it offline: the Space boots by memory-mapping files that are already
# there, instead of downloading 1.4GB of parquet and spending 50 minutes
# embedding on a 2-vCPU container before it can serve a single request.
set -euo pipefail

USER="${1:?usage: push_to_space.sh <hf-username> <space-name>}"
SPACE="${2:?usage: push_to_space.sh <hf-username> <space-name>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"

command -v git-lfs >/dev/null || { echo "git-lfs required: brew install git-lfs"; exit 1; }
[ -f "$REPO_ROOT/data/index/manifest.json" ] || {
  echo "No index found. Run: uv run python -m vrag.build_index"; exit 1; }

echo "→ staging in $STAGING"
git clone "https://huggingface.co/spaces/$USER/$SPACE" "$STAGING" 2>/dev/null \
  || { mkdir -p "$STAGING" && git -C "$STAGING" init -q; }

cd "$STAGING"
git lfs install --local

# The Space's README front-matter is its configuration — sdk, port, title.
cp "$REPO_ROOT/deploy/README_SPACE.md" README.md
cp "$REPO_ROOT/app.py" "$REPO_ROOT/requirements.txt" "$REPO_ROOT/pyproject.toml" .
rm -rf src data scripts && mkdir -p data
cp -r "$REPO_ROOT/src" .
cp -r "$REPO_ROOT/scripts" .
cp -r "$REPO_ROOT/data/index" data/index

cat > .gitattributes <<'EOF'
*.usearch filter=lfs diff=lfs merge=lfs -text
*.arrow   filter=lfs diff=lfs merge=lfs -text
*.npz     filter=lfs diff=lfs merge=lfs -text
data/index/bm25/** filter=lfs diff=lfs merge=lfs -text
EOF

# Belt and braces: .env is gitignored in the source repo, but this script
# copies directories wholesale into a fresh repo with its own ignore rules.
# An API key reaching a public Space is not a recoverable mistake.
rm -f .env .env.local
printf '.env\n.env.local\n__pycache__/\n' > .gitignore

git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "Deploy voice-rag" || echo "(nothing changed)"

echo "→ pushing to https://huggingface.co/spaces/$USER/$SPACE"
echo "  Use an HF access token with WRITE scope as the password."
git push -f https://huggingface.co/spaces/"$USER"/"$SPACE" HEAD:main

cat <<EOF

Done. Next:
  1. Space -> Settings -> Variables and secrets -> add SARVAM_API_KEY
     (and ANTHROPIC_API_KEY if you want the LLM answer path)
     Without SARVAM_API_KEY the microphone is disabled; text still works.
  2. Watch the build logs, then open:
     https://huggingface.co/spaces/$USER/$SPACE
  3. Verify on the deployed box (2 vCPU, not your laptop):
     the per-stage table should still show pipeline < 200ms.
EOF
