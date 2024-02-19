#!/bin/bash
GPUS_PER_NODE=4
mkdir -p ${HOME}/hostfiles
# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile=${HOME}/hostfiles/hosts_$SLURM_JOBID
# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done