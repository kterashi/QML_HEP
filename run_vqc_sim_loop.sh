#!/bin/bash


njob=30
nevt=(20)
niter=(100)
ndepth_uin=(1)
ndepth_uvar=(1)
nqubit=(3 5 7)

feat_map="FirstOrder"
#feat_map="SecondOrder"

bkend="ibmq_johannesburg"
#bkend="ibmq_boeblingen"

opt_="def_q012"


fm=""
if [ ${feat_map} = "SecondOrder" ]; then
    fm="o2"
elif [ ${feat_map} = "FirstOrder" ]; then
    fm="o1"
fi

for evt in ${nevt[@]}; do
    for iter in ${niter[@]}; do
	for depth_uin in ${ndepth_uin[@]}; do
	    for depth_uvar in ${ndepth_uvar[@]}; do
		for nq in ${nqubit[@]}; do
		    for ijob in $(seq ${njob}); do
			opt=${opt_}_run${ijob}
			sed s/NEVT/$evt/ vqc_sim.py | sed s/NQUBIT/${nq}/ | sed s/NITER/$iter/ | sed s/FEATMAP/${feat_map}/ | sed s/FMAP/${fm}/ | sed s/NDEPTH_UIN/$depth_uin/ | sed s/NDEPTH_UVAR/$depth_uvar/ | sed s/BACKENDNAME/$bkend/ | sed s/OPTION/$opt/ | sed s/JOBN/${ijob}/ > ./vqc_sim_${nq}d_${evt}evt_iter${iter}_${fm}uin-depth${depth_uin}_uvar-depth${depth_uvar}_${bkend}_${opt}.py
			log=vqc_sim_${nq}d_${evt}evt_iter${iter}_${fm}uin-depth${depth_uin}_uvar-depth${depth_uvar}_${bkend}_${opt}.log
			python ./vqc_sim_${nq}d_${evt}evt_iter${iter}_${fm}uin-depth${depth_uin}_uvar-depth${depth_uvar}_${bkend}_${opt}.py >& ${log}
		    done
		done
	    done
	done
    done
done