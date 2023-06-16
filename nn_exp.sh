#!/bin/bash
# Iteratively runs nn_test.py after incrementing # of layers

for i in {1..6}
do
    python3 nn_test.py $i >> experiment.txt
done
