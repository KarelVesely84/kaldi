#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: 2022-2024 Karel Vesely (iveselyk@fit.vutbr.cz)

import logging
import os

import numpy as np

import pywrapfst as fst  # see PYTHONPATH in `run_example.sh`


class BoostingFstBuilder:
    """
    This class builds fsts for lattice-boosting and hclg boosting.

    The intended fst composition is with RhoMatcher matcher:
    https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#Matchers

    The RhoMatcher "consumes the symbol" and "matches rest", which deserves
    to be explained practically later. The lattice boosting is a composition:

    L' = L o B

    where L is lattice, and B is boosting graph. Suppose the composition alg.
    is in two states, one in lattice L and one in boosting graph B.
    The lattice state L(s) has many outgoing arcs, and the boosting state B(s)
    has only a few outgoing arcs and one of them has ρ (rho) as the input and output symbol.

    This ρ arc acts as a wildcard, so if the outgoing arc of L(s) cannot be matched
    with any outgoing arc of B(s), the arc from L(s) and the ρ arc from B(s) are
    traversed jointly. And because ρ being the output symbol, the output symbol
    from lattice L is forwarded into the boosted lattice L'.

    In paper 'Boosting of contextual information in ASR for air-traffic cann-sign recognition',
    we were originally using 'phi' matching, while we included whole vocabulary into the
    boosting graph. This is no longer needed thanks to using 'rho' matching.

    """

    def __init__(self, words_txt: str):
        # load words.txt into a map,
        self.word_to_int = {
            w: i for w, i in np.loadtxt(words_txt, dtype="object,i4", comments=None)
        }

        # remove epsilon, it's always '0',
        del self.word_to_int["<eps>"]

        # store 'rho' symbol for 'rho'-composition, and remove it from the dict,
        self.rho_int = self.word_to_int["#0"]
        del self.word_to_int["#0"]

    def lattice_boosting_graph(
        self,
        boosted_phrases: list[list[str]],
        word_discount: float = -3.0,
        phrase_discount: float = 0.0,
    ) -> fst.VectorFst:  # TODO: exact type ?
        """
        Build lattice-boosting graph for the particular set of boosting phrases.

        Args:
            boosted_phrases: tokenized text to be boosted,
                             outer 'list' -> M phrases to be boostedm
                             inner 'list' -> N words in 1 boosting phrase, score discount given only
                                             if all words in the phrase match in the lattice
            word_discount: score discount added to lattice-arc weight per each word of the boosting phrase
                           (implemented as cummulative discount on last word of that phrase).
            phrase_discount: score discount of boosting phrase, regardless of its length
                             (but using just word_discount was satisfactory).

        Returns:
            fst.VectorFst
        """

        # build non-optimized graph
        f = BoostingFstBuilder.__gen_initial_graph(
            boosted_phrases, self.word_to_int, word_discount, phrase_discount
        )

        f = fst.determinize(f)

        # point all states to initial state by <eps> link with zero cost, make them non-final,
        # - note: the score discount of the whole boosted word sequence is given to its last word
        s_start = f.start()
        for s in range(f.num_states()):
            if s != s_start:
                # <eps> link to initial state with zero cost,
                f.add_arc(s, fst.Arc(ilabel=0, olabel=0, weight=0.0, nextstate=s_start))
                # make final states non-final,
                if float(f.final(s)) != float("inf"):
                    f.set_final(s, float("inf"))

        # only the initial state is final state,
        f.set_final(s_start, 0.0)

        # add the 'rho' link to initial state
        # - it matches the non-boosted part of graph
        # - 'rho' is both input and output symbol
        rho = self.rho_int
        f.add_arc(
            s_start, fst.Arc(ilabel=rho, olabel=rho, weight=0.0, nextstate=s_start)
        )

        return f

    @staticmethod
    def __gen_initial_graph(
        boosted_phrases, word_to_int, word_discount, phrase_discount
    ):
        """
        Generate boosting graph that is not yet optimized (it is larger).
        """

        f = fst.VectorFst()
        s_start = f.add_state()
        f.set_start(s_start)

        for boosted_phrase in boosted_phrases:

            # convert vector<str> to vector<int>,
            try:
                boosted_phrase_ints = [word_to_int[w] for w in boosted_phrase]
            except KeyError:
                oovs = [w for w in boosted_phrase if w not in set(word_to_int.keys())]
                logging.warning(
                    f"The words `{oovs}` are not present in 'words.txt'. It is an Out-of-vocabulary word that cannot be boosted, skipping boosted phrase {boosted_phrase}"
                )
                continue

            if len(boosted_phrase_ints) == 0:
                continue

            # split to prefix and last word,
            *w_rest, w_last = boosted_phrase_ints

            # zero weight here,
            s_prev = s_start
            for w in w_rest:
                s_next = f.add_state()
                f.add_arc(
                    s_prev, fst.Arc(ilabel=w, olabel=w, weight=0.0, nextstate=s_next)
                )
                s_prev = s_next

            # weight for whole boosted word sequence is 'sitting' on its last word,
            # - previous words will have 'eps' links to the initial state, only initial state has the 'rho' arc
            weight = len(boosted_phrase_ints) * word_discount + phrase_discount

            s_next = f.add_state()
            f.add_arc(
                s_prev,
                fst.Arc(ilabel=w_last, olabel=w_last, weight=weight, nextstate=s_next),
            )

            # set final state,
            f.set_final(s_next, 0.0)

        return f


def main():
    """
    This main() funtion contains Example code for 'eyeball' testing.
    """

    # prepare words.txt into a tmpfile,
    import tempfile

    words_fd, words_fname = tempfile.mkstemp(prefix="pylet_")
    os.write(
        words_fd,
        """<eps> 0
c_s_a 1
one 2
two 3
three 4
alfa 5
bravo 6
brno 7
prague 8
#0 9
""".encode(),
    )

    boosted_phrases = [["c_s_a", "three", "alfa", "bravo"], ["c_s_a", "alfa", "bravo"]]

    builder = BoostingFstBuilder(words_fname)

    f = builder.lattice_boosting_graph(
        boosted_phrases, word_discount=-4.0, phrase_discount=-2.0
    )

    f.write("example_lattice_boosting_graph.fst")

    print("%%% Latice boosting graph")
    print(f.print())

    # visualisation: this will create .dot and .pdf files
    if True:
        # read symbol table from text file,
        symbol_table = fst.SymbolTable().read_text(words_fname)

        fname = "example_lattice_boosting_graph"  # .dot .pdf
        f2 = f.copy()
        f2.project("input")  # keep just the input symbols
        # call fstdraw,
        f2.draw("%s.dot" % fname, isymbols=symbol_table, acceptor=True)
        # create pdf,
        os.system(
            "cat %s.dot | grep -v '^orientation =' | dot -Tpdf >%s.pdf" % (fname, fname)
        )


if __name__ == "__main__":
    main()
