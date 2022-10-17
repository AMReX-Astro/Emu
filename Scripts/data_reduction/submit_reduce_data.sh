#!/bin/bash
#
# Number of nodes:
#SBATCH --nodes=8
#
#################
# Haswell nodes
#################
#
# Requests Cori Haswell nodes:
#SBATCH --constraint=haswell
#
# Haswell: Assign 1 MPI task to each socket
#SBATCH --tasks-per-node=16
#
# Haswell: each socket has 32 CPUs (with hyperthreading)
#SBATCH --cpus-per-task=4
#
#################
# Queue & Job
#################
#
# Which queue to run in: debug, regular, premium, etc. ...
#SBATCH --qos=regular
#
# Run for this much walltime: hh:mm:ss
#SBATCH --time=24:00:00
#
# Use this job name:
#SBATCH -J emu_reduce
#
# Send notification emails here:
#SBATCH --mail-user=srichers@berkeley.edu
#SBATCH --mail-type=ALL
#
# Which allocation to use:
#SBATCH -A m3761

# On the compute node, change to the directory we submitted from
cd $SLURM_SUBMIT_DIR

module load python3

srun -n 128 -c 4  python3 ~/emu_scripts/data_reduction/reduce_data.py
