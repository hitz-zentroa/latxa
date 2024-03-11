#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00         # format: HH:MM:SS
#SBATCH --nodes 8               # 4 node
#SBATCH --ntasks-per-node=4     # 4 tasks out of 32
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB              # memory per node out of 494000MB
#SBATCH --output=.slurm/latxa-7b-v1.1.out
#SBATCH --error=.slurm/latxa-7b-v1.1.err
#SBATCH --exclusive
#SBATCH --requeue

#--open-mode=append

# setup the virtual env
source ${WORK}/environments/neox-env/bin/activate

# load leonardo modules
module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs

ds_report

# set distributed env variable flags such as NCCL_DEBUG here

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

# Write the hostfile for this job here
# Should write to a hostfile that contains lines of format `<machine IP> slots=<NUM_GPUS_PER_NODE>`
bash ${HOME}/llama-eus/leonardo/train/write_hostfile.sh
export DLTS_HOSTFILE=${HOME}/hostfiles/hosts_$SLURM_JOBID

# Model configuration
TP=1
PP=0

# Prepare to start finetuning or continue from checkpoint
SAVE_PATH=/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b-neox-eus-v1.1_500k
if [ -d "$SAVE_PATH" ]; then
    LOAD_PATH=$SAVE_PATH
    FINETUNE="false"
    CONF_NAME="continue"
    printf "Continue training from %s\n" $SAVE_PATH
else
    LOAD_PATH=/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b-neox-TP-${TP}
    FINETUNE="true"
    CONF_NAME="start"
    printf 'Starting training from %s\n' ${LOAD_PATH}
fi

printf '{\n "save": "%s",\n "load": "%s",\n "finetune": %s\n}' ${SAVE_PATH} ${LOAD_PATH} $FINETUNE \
    >${HOME}/llama-eus/leonardo/configs/save_load/${CONF_NAME}_7B_eus_v1.1_500k.yml

# Move to the gpt-neox install
#TRAIN_PATH=/leonardo_scratch/large/userexternal/jetxaniz/gpt-neox/
TRAIN_PATH=${WORK}/gpt-neox
cd $TRAIN_PATH

# launch distributed job. If using `"deepspeed_slurm": true` and `"launcher": "slurm"` on a SLURM cluster,
# then NeoX will handle the creation of a distributed run across gpus.
python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
    --conf_dir ${HOME}/llama-eus/leonardo/configs \
    base_config.yml \
    models/llama-2-7b.yml \
    deepspeed/zero1.yml \
    hyperparameters/7B_v1.1.yml \
    data/latxa-v1.1.yml \
    save_load/${CONF_NAME}_7B_eus_v1.1_500k.yml
