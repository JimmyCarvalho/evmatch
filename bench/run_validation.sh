#!/bin/sh
# Headline validation: sinter, 1e6 shots per point, then the belief-matching reference (paired, fewer shots).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
echo "validation start $(date -u +%FT%TZ)" >> logs/compute.log
$PY bench/collect.py --decoders evmatch nat-single-correlated --distances 5 7 9 \
    --ps 0.001 0.002 0.003 0.004 0.005 --max-shots 1000000 --save results/main.csv > logs/main.log 2>&1
echo "validation main end $(date -u +%FT%TZ)" >> logs/compute.log
$PY bench/paired.py --decoders evmatch pymatching-correlated beliefmatching --d 5 --p 0.003 0.005 --shots 100000 --chunk 2500 --jsonl results/bm_reference.jsonl > logs/bm_reference.log 2>&1
$PY bench/paired.py --decoders evmatch pymatching-correlated beliefmatching --d 7 --p 0.003 0.005 --shots 50000 --chunk 1250 --jsonl results/bm_reference.jsonl >> logs/bm_reference.log 2>&1
$PY bench/paired.py --decoders evmatch pymatching-correlated beliefmatching --d 9 --p 0.003 0.005 --shots 25000 --chunk 625 --jsonl results/bm_reference.jsonl >> logs/bm_reference.log 2>&1
echo "validation end $(date -u +%FT%TZ)" >> logs/compute.log
