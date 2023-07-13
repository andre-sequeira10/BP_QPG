#!/bin/bash

N=5
for j in $(seq 1 30)
do
	((i=i%N)); ((i++==0)) && wait
	python qnpg_softmax_TRUE_product_approx.py --init glorot --policy Q --ng 0 --n_layers 3 --episodes 500 --entanglement all2all --batch_size 10 --local_measurements 1 --inpt_scaling 0 &
done



