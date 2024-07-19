#!/bin/bash

./configure --shared --mathlib=MKL --use-cuda=no --with-cudadecoder=no

# make -j clean depend && make -j 4
