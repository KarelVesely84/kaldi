#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: 2020-2024 Karel Vesely (iveselyk@fit.vutbr.cz)

import argparse

from boosting_fst_lib import BoostingFstBuilder

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--word-discount", default=-3.0, type=float, help="Word boosting constant.")
    parser.add_argument("--phrase-discount", default=0.0, type=float, help="Phrase boosting constant.")
    parser.add_argument("words_txt", help="words.txt file")
    parser.add_argument("boosted_phrases", help="boosted phrases, one phrase per line")
    parser.add_argument("fst_out", help="output fst for lattice boosting")

    return parser.parse_args()

def main():
    args = parse_args()

    builder = BoostingFstBuilder(args.words_txt)

    # load & tokenize boosted_text
    with open(args.boosted_phrases, mode="r") as f:
        boosted_phrases = f.readlines()
    boosted_phrases = [ line.strip().split() for line in boosted_phrases ]

    fst_ = builder.lattice_boosting_graph(
        boosted_phrases,
        word_discount=args.word_discount,
        phrase_discount=args.phrase_discount,
    )

    fst_.write(args.fst_out)

if __name__ == "__main__":
    main()
