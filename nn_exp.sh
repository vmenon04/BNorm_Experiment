#!/bin/bash
# Iteratively runs nn_test.py after incrementing # of layers

total_start=`date +%s.%N`

echo Layer,Raw Precision,BNorm Precision,Raw Recall,BNorm Recall,Raw Time,BNorm Time > interaction_relu.csv
for i in {1..6}
do
    echo Num_Layers=$i

    start=`date +%s.%N`

    python3 nn_test.py $i relu >> interaction_relu.csv
 
    end=`date +%s.%N`

    runtime=$( echo "$end - $start" | bc -l )
    echo Time: $runtime

done

echo Layer,Raw Precision,BNorm Precision,Raw Recall,BNorm Recall,Raw Time,BNorm Time > interaction_tanh.csv
for i in {1..6}
do
    echo Num_Layers=$i

    start=`date +%s.%N`

    python3 nn_test.py $i tanh >> interaction_tanh.csv
 
    end=`date +%s.%N`

    runtime=$( echo "$end - $start" | bc -l )
    echo Time: $runtime

done

echo Layer,Raw Precision,BNorm Precision,Raw Recall,BNorm Recall,Raw Time,BNorm Time > interaction_sigmoid.csv
for i in {1..6}
do
    echo Num_Layers=$i

    start=`date +%s.%N`

    python3 nn_test.py $i sigmoid >> interaction_sigmoid.csv
 
    end=`date +%s.%N`

    runtime=$( echo "$end - $start" | bc -l )
    echo Time: $runtime

done

total_end=`date +%s.%N`
total_runtime=$( echo "$total_end - $total_start" | bc -l )

python3 notify.py "${total_runtime}"