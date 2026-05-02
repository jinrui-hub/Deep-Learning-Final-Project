#!/bin/bash
#SBATCH -J exaperiment1_seperateJudge              # job name
#SBATCH -o logs/job_%j.out      # stdout log file (%j is replaced by job ID)
#SBATCH -A ASC25003
#SBATCH -p gpu-a100-small       # partition (1x A100)
#SBATCH -N 1                    # number of nodes
#SBATCH -n 1                    # number of tasks
#SBATCH -t 08:00:00             # max wall time
#SBATCH --mail-user=jinrui@utexas.edu   # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # notify on job start, end, or failure

cd /work/11251/jinrui/ls6/LLMasRNN

conda init
conda activate lost             # activate conda environment

export PORTKEY_API_KEY=j2gdRq2PrdQzJ7wo8+ectwiMG8Tq

python run_experiment.py \
  --config configs/exaperiment1_seperateJudge.yaml \
  --run-name exaperiment1_seperateJudge \
  --methods llm_as_rnn
