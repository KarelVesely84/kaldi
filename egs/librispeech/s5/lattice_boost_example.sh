#!/bin/bash

. path.sh
. cmd.sh

# assuming the models were downloaded and extracted from:
# https://kaldi-asr.org/models/m13

# assuming librispeech is downloaded and prepared: run.sh, stages 1-3

set -eux

# extract hires mfcc features
datadir=dev_other
false && \
{
    utils/copy_data_dir.sh data/${datadir}{,_hires}
    steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
}

# extract ivector features
data=${datadir}
false && \
steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    data/${data}_hires exp/nnet3_cleaned/extractor \
    exp/nnet3_cleaned/ivectors_${data}_hires

# build HCLG.fst
lang_test=data/lang_test_tgsmall
dir=exp/chain_cleaned/tdnn_1d_sp
graph_dir=${dir}/graph_tgsmall
false && \
utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov ${lang_test} ${dir} ${graph_dir}

# generate lattices
decode_nj=20
decode_set=dev_other
false && \
steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $decode_nj --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${decode_set}_hires \
    $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_tgsmall

# build the boosting graphs
# hint: run as ( SUBPROCESS ), so it does not mess up env for other tools
#false && \
(
    # ENABLE `pywrapfst` PATHON API OF OpenFST:
    python=$(basename $(readlink -m $(which python)))
    pywrapfst=$KALDI_ROOT/tools/openfst/lib/${python}/site-packages
    export PYTHONPATH=$pywrapfst
    libpython_dir=$(dirname $(which python))/../lib  # relative path to conda env.
    export LD_LIBRARY_PATH=$libpython_dir

    words_txt=data/lang_nosp/words.txt
    boosted_phrases_ark=dev_other_latboost.phrases
    fst_out_ark=dev_other_latboost.ark
    # create boosted phrases from correct transcripts
    # - a boosted phrase is reference text splitted into 5-word chunks (splitted by '|', there's no overlap across chunks)
    cat data/dev_other/text | awk '{ for(i=7; i<NF; i+=5) { $(i) = "| "$(i); } print $0; }' >${boosted_phrases_ark}
    # create boosting graphs
    ./lattice_boosting/make_ark_of_boosting_graphs.py --word-discount -3.0 ${words_txt} ${boosted_phrases_ark} ${fst_out_ark}
    # => this will show warnings for OOVs in the boosted phrases...
)

# boost the lattices, get WER
lang_or_graph=data/lang_nosp_test_tgsmall
data=data/dev_other_hires
lat_dir_in=exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall
boosting_graphs=ark:dev_other_latboost.ark
lat_dir_out=exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall_latboost
./lattice_boosting/rescore_lattices.sh ${lang_or_graph} ${data} ${lat_dir_in} ${boosting_graphs} ${lat_dir_out}

# show WER scores
#
# no-boost:
cat exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall/wer_* | ./utils/best_wer.sh
# %WER 12.25 [ 6243 / 50948, 541 ins, 839 del, 4863 sub ]
#
# latboost:
cat exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall_latboost/wer_* | ./utils/best_wer.sh
# %WER 4.05 [ 2064 / 50948, 224 ins, 276 del, 1564 sub ]
