#!/bin/sh
# Builds the arXiv upload (source + figures) and checks that it compiles.
set -e
cd "$(dirname "$0")"
cp ../results/ler.png ../results/ratio.png .
pdflatex -interaction=nonstopmode evmatch.tex > /dev/null
pdflatex -interaction=nonstopmode evmatch.tex > /dev/null
tar czf arxiv_evmatch.tar.gz evmatch.tex ler.png ratio.png
echo "paper/arxiv_evmatch.tar.gz"
