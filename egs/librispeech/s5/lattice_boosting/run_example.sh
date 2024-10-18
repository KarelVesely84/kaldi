#!/bin/bash

set -euxo pipefail

KALDI_ROOT=$PWD/../../../..

# MAKE OPENFST ACCESSIBLE FROM PYTHON
python=$(basename $(readlink -m $(which python)))
pywrapfst=$KALDI_ROOT/tools/openfst/lib/${python}/site-packages
# ${pywrapfst} contains files: pywrapfst.a, pywrapfst.la, pywrapfst.so
# make sure openfst was copiled with `--enable-python` in `OPENFST_CONFIGURE` at `tools/Makefile`
export PYTHONPATH=$pywrapfst

libpython_dir=$(dirname $(which python))/../lib  # relative path to conda env.
export LD_LIBRARY_PATH=$libpython_dir

words_txt=../data/lang_nosp/words.txt
boosted_phrases=boosted_phrases.txt
fst_out=boosted_phrases.fst
./make_sigle_boosting_graph.py --word-discount -3.0 ${words_txt} ${boosted_phrases} ${fst_out}

# not yet ready
# ./make_ark_of_boosting_graphs.py
