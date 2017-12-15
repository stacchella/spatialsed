#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'spatial_test'
### output and error logs
#SBATCH -o stest.out
#SBATCH -e stest.err
source activate pro
srun -n 1 python ../../prospector/scripts/prospector_dynesty.py \
--param_file=demo_mock_params.py



