#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=test_idx_load
#SBATCH --output=/home/%u/logs/test_index_load-%j.out
#
# Isolated diagnostic: times the ACL vector index load and a single
# retrieval call. Nothing else runs — no agent, no API. Run when
# eval_scico jobs hang silently after `Loading weights` to pinpoint
# whether the bottleneck is index I/O over NFS.
#
# Expected outputs and what they mean:
#   "load_acl_index returned" within 30s   -> index load is fine
#   "load_acl_index returned" after 5-15min -> NFS is slow, stage to /tmp
#   stuck on "calling load_acl_index"       -> NFS hang, needs intervention

set -euo pipefail

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

if [ -f .env ]; then set -a; . .env; set +a; fi

python3 -u -c "
import os, time
t0 = time.time()
def stamp(s): print(f'[{time.time()-t0:6.1f}s] {s}', flush=True)

stamp('starting')
from backend.rag.local_acl_indexing import load_acl_index, DEFAULT_ACL_PERSIST_DIR
stamp(f'imports done')
stamp(f'persist_dir = {DEFAULT_ACL_PERSIST_DIR}')

# Index size (informational)
total_size = 0
n_files = 0
for root, _, files in os.walk(DEFAULT_ACL_PERSIST_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))
        n_files += 1
stamp(f'index dir: {n_files} files, {total_size / 1e9:.2f} GB total')

stamp('calling load_acl_index ...')
idx = load_acl_index(DEFAULT_ACL_PERSIST_DIR)
stamp(f'load_acl_index returned; docs={len(idx.docstore.docs)}')

stamp('embedding a test query (warms MiniLM, exercises the retriever)')
r = idx.as_retriever(similarity_top_k=5).retrieve('retrieval augmented generation')
stamp(f'retrieved {len(r)} nodes')
for n in r[:2]:
    md = getattr(n, 'metadata', None) or getattr(n.node, 'metadata', {})
    stamp(f'  - paper_id={md.get(\"paper_id\")} score={n.score:.3f}')

stamp('all done')
"
