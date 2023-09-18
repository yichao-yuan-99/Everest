#!/bin/bash

sudo /usr/local/cuda-11.4/bin/ncu --target-processes all --section WarpStateStats --section SpeedOfLight --section Occupancy --section InstructionStats ./run-legacy.sh ../inputs/graphs/wiki-talk-temporal ../inputs/motifs/M6.txt 86400 Baseline | tee .tmp/prof_wiki_m6_base.out
sudo /usr/local/cuda-11.4/bin/ncu --target-processes all --section WarpStateStats --section SpeedOfLight --section Occupancy --section InstructionStats ./run-legacy.sh ../inputs/graphs/wiki-talk-temporal ../inputs/motifs/M6.txt 86400 U1B | tee .tmp/prof_wiki_m6_u1b.out
sudo /usr/local/cuda-11.4/bin/ncu --target-processes all --section WarpStateStats --section SpeedOfLight --section Occupancy --section InstructionStats python3 single-query.py queries/prof/WI-M6.yaml | tee .tmp/prof_wiki_m6_full.out