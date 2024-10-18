#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: 2020-2024 Karel Vesely (iveselyk@fit.vutbr.cz)

import argparse
import os
import sys
import numpy as np
import pywrapfst as fst

from boosting_fst_lib import BoostingFstBuilder

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--word-discount",
        default=-3.0,
        metavar='<real>',
        type=float,
        help="Per-word score boosting discount.",
    )
    parser.add_argument(
        "--phrase-discount",
        default=0.0,
        metavar='<real>',
        type=float,
        help="Per-phrase score boosting discount.",
    )
    parser.add_argument(
        "words_txt",
        help="Openfst word symbol table (e.g. lang/words.txt)",
    )
    parser.add_argument(
        "boosted_phrases_ark",
        help="Ark file with text input containing boosted phrases separated by '|'.",
    )
    parser.add_argument(
        "fst_out_ark",
        help="Ark file for storing the boosting graphs.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    builder = BoostingFstBuilder(args.words_txt)

    # load boosted_text
    with open(args.boosted_phrases_ark, mode="r") as f:
        boosted_phrases = f.readlines()

    with open(args.fst_out_ark, mode="wb") as f:
        for utt_boost_line in boosted_phrases:
            # separate utt_key
            utt_key, boosted_phrases = utt_boost_line.strip().split(maxsplit=1)
            # split phrases, tokenize
            boosted_phrases = [ phrase.split() for phrase in boosted_phrases.split('|') ]

            fst_ = builder.lattice_boosting_graph(
                boosted_phrases,
                word_discount=args.word_discount,
                phrase_discount=args.phrase_discount,
            )

            # store the fst in ark,
            f.write((utt_key+" ").encode("utf8"))
            f.write(fst_.write_to_string())

if __name__ == "__main__":
    main()
