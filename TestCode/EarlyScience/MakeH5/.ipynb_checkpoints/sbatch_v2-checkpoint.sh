#!/bin/bash

#source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh #this sets environment for psana2. note psana2 not compatible with psana(1)
#base_path=/cds/home/d/dgarratt/Code2020/chemRIXS/preproc/v1
base_path=/cds/home/n/npowersr/XAS/ChemRIXS/rixlw1019/results/Dougie/preproc/v2
script=$base_path/preproc_v2.py
log=/reg/data/ana16/rix/rixlw1019/results/preproc/v2/logs/run$1_v2.log

if [ -z "$2" ]
then
    n_nodes=3
else
    n_nodes=$2
fi

if [ -z "$3" ]
then
    tasks_per_node=16 #I think there are 16 procs per node for the feh queues, 64 for the ffb and 12 for the 'old' psana qs
else
    tasks_per_node=$3
fi

echo $log
echo $script
# psfehq, psfehprioq, psfehprioq, psfehhiprioq, psanaq
sbatch -p psfehhiprioq -N $n_nodes -n $tasks_per_node --output $log --wrap="mpirun python $script $1"