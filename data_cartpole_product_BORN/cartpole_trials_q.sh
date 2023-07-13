#!/bin/bash

N=5
for j in $(seq 1 15)
do
	((i=i%N)); ((i++==0)) && wait
	python qnpg_born_TRUE_product_approx.py --init glorot --policy Q --ng 0 --n_layers 5 --episodes 500 --entanglement all2all --batch_size 10 &
done

#N=5
#for j in $(seq 1 20)
#do
	#((i=i%N)); ((i++==0)) && wait
	#python qnpg_reinforce_cartpole.py --init glorot --policy Q --ng 0 --n_layers 4 --episodes 1000 --entanglement all2all --batch_size 12 &
#done


