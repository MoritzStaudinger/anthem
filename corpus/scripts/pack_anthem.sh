#!/bin/bash
# Pack the ANTHEM source trees into zstd archives for transfer off the cluster.
#
# Run this ON THE CLUSTER (login node or a JupyterHub terminal). It tars each
# source directory separately so a failure only costs one archive, and prints a
# size listing at the end.
#
# Transferring archives rather than loose files matters: the trees hold ~250k
# small files, so loose rsync would mean a quarter of a million round-trips.
# acl_pdfs_parsed really is ~84 GB (121,288 TEI files averaging ~690 KB), and
# compresses about 3x, so budget ~30 GB for that archive alone.
#
# The output must NOT go to $HOME: home is under a disk quota that an archive
# this size blows through, and tar/zstd die mid-write with
# "Disk quota exceeded". Default OUT is therefore on /share.
#
# Usage:
#   nohup ./pack_anthem.sh > /share/moritz.staudinger/anthem_pack.log 2>&1 &
#   tail -f /share/moritz.staudinger/anthem_pack.log
#
# nohup matters in JupyterHub: packing takes hours and an idled-out terminal
# would otherwise kill it.
#
# Override the defaults with environment variables:
#   SRC=/share/moritz.staudinger/acl_data OUT=/share/moritz.staudinger/pack ./pack_anthem.sh
#
# Then, from the local machine:
#   rsync -avP moritz.staudinger@cluster.datalab.tuwien.ac.at:/share/moritz.staudinger/pack/'anthem-*.tar.zst' .
#   for f in anthem-*.tar.zst; do tar -I zstd -xf "$f"; done

SRC="${SRC:-/share/moritz.staudinger/acl_data}"
OUT="${OUT:-/share/moritz.staudinger/anthem_pack}"

cd "$SRC" || { echo "ERROR: cannot cd to $SRC"; exit 1; }
command -v zstd >/dev/null || { echo "ERROR: zstd not found; use tar -z and .tar.gz"; exit 1; }
mkdir -p "$OUT" || { echo "ERROR: cannot create $OUT"; exit 1; }

# Refuse to write into a quota-limited home: a ~30 GB archive will not fit and
# tar dies partway through, leaving a useless file that still occupies quota.
case "$OUT" in
  "$HOME"|"$HOME"/*)
    echo "ERROR: OUT is under \$HOME, which is quota-limited. Use a path on /share."
    exit 1 ;;
esac

AVAIL_GB=$(df -BG --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 60 ]; then
  echo "WARNING: only ${AVAIL_GB} GB free at $OUT; the parse trees need ~45 GB of archives."
fi

# One archive each: the parse trees plus the derived layers the pipeline produced.
BIG="acl_pdfs_parsed acl_pdfs_pymu_llm acl_pdfs_hybrid ror_parsed_affiliations_90_with_topics_and_bibmeta gemini_affiliations s2orc_normalized s2orc_raw"

# Small intermediate generations, bundled together.
MISC="ror_parsed_affiliations ror_parsed_affiliations_90 ror_parsed_affiliations_90_with_topics ror_parsed_affiliations_organization_type90 cleaned_affiliations_90 cleaned_affiliations_98 acl_affiliations_json acl_lists s2orc_affiliations"

echo "# source : $SRC"
echo "# output : $OUT"
echo "# started: $(date -Is)"

for d in $BIG; do
  if [ -d "$d" ]; then
    if tar -I "zstd -10 -T0" -cf "$OUT/anthem-$d.tar.zst" "$d"; then
      echo "OK   $d"
    else
      echo "FAIL $d"
    fi
  else
    echo "SKIP $d (missing)"
  fi
done

EXIST=""
for d in $MISC; do
  [ -d "$d" ] && EXIST="$EXIST $d"
done
if [ -n "$EXIST" ]; then
  if tar -I "zstd -10 -T0" -cf "$OUT/anthem-misc.tar.zst" $EXIST; then
    echo "OK   misc:$EXIST"
  else
    echo "FAIL misc"
  fi
else
  echo "SKIP misc (nothing present)"
fi

echo "# finished: $(date -Is)"
echo ALLDONE
ls -lh "$OUT"/anthem-*.tar.zst
