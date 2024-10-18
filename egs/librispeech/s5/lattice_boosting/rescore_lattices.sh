#!/usr/bin/env bash

# Copyright 2024  Karel Vesely
# Apache 2.0.

# This script does lattice boosting for pre-generated lattices.

# Begin configuration section.
stage=1
cmd=run.pl
scoring_opts="--min-lmwt 12 --max-lmwt 17 --word-ins-penalty 0.0"
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <lang-dir> <data-dir> <lat-dir-in> <boosting-graphs> <lat-dir-out>"
  echo "e.g.:   lattice_boosting/rescore_lattices.sh \\"
  echo "    data/lang_nosp data/dev_other_hires \$dir/decode_dev_other_tgsmall \\"
  echo "    ark:dev_other_latboost.ark \$dir/decode_dev_other_tgsmall_latboost"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --skip-scoring <bool>                    # whether to skip local/score.sh"
  exit 1;
fi

lang_or_graph=$1
data=$2
lat_dir_in=$3
boosting_graphs=$4
lat_dir_out=$5

# Example arguments:
# lang_or_graph=data/lang_nosp_test_tgsmall
# data=data/dev_other_hires
# lat_dir_in=exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall
# boosting_graphs=ark:lattice_boosting/dev_other_latboost.ark
# lat_dir_out=exp/chain_cleaned/tdnn_1d_sp/decode_dev_other_tgsmall_latboost

for f in $lang_or_graph/words.txt $data/text $lat_dir_in/lat.1.gz ${boosting_graphs#ark:}; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

rho_sym=$(grep '#0' data/lang_nosp/words.txt | awk '{ print $2;}')
nj=$(cat ${lat_dir_in}/num_jobs)

mkdir -p ${lat_dir_out}

echo "lattice-compose"

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $lat_dir_out/log/rescore_lattices.JOB.log \
      lattice-compose --rho-label=${rho_sym} --compose-with-fst=true \
      "ark:gunzip -c ${lat_dir_in}/lat.JOB.gz |" \
      ${boosting_graphs} \
      "ark:| gzip -c >${lat_dir_out}/lat.JOB.gz" || exit 1;
fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 2 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    local/score.sh ${scoring_opts} --cmd "${cmd}" ${data} ${lang_or_graph} ${lat_dir_out} || exit 1;
    echo "score confidence and timing with sclite"
  fi
fi
echo "Lattice boosting done."
exit 0;
