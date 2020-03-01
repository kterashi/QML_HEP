#!/bin/bash

njob=100
uin_type=(0)
nevt=(100 1000)
niter=(50000)
ndepth=(3)
nqubit=(3 5 7)

opt="COBYLA_def"

for uin in ${uin_type[@]}; do
    for evt in ${nevt[@]}; do
	for iter in ${niter[@]}; do
	    for depth in ${ndepth[@]}; do
		for qubit in ${nqubit[@]}; do
		    for ijob in $(seq ${njob}); do
			rnum=`date "+%-m%d%H%M%S"`
			sed s/NEVT/$evt/ qulacs_qcl.py | sed s/UIN/$uin/ | sed s/NITER/$iter/ | sed s/NDEPTH/$depth/ | sed s/OPTION/$opt/ | sed s/NVAR/${qubit}/ | sed s/JOBN/${ijob}/ | sed s/RNUM/${rnum}/ > ./qulacs_qcl_uin${uin}_${qubit}d_${evt}evt_iter${iter}_depth${depth}_${opt}_run${ijob}.py
			log=qulacs_qcl_uin${uin}_${qubit}d_${evt}evt_iter${iter}_depth${depth}_${opt}_run${ijob}.log
			python ./qulacs_qcl_uin${uin}_${qubit}d_${evt}evt_iter${iter}_depth${depth}_${opt}_run${ijob}.py >& ${log}
		    done
		done
	    done
	done
    done
done
