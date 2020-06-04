#!/bin/bash
#
# Number of nodes:
#SBATCH --nodes=1
#
#################
# GPU nodes
#################
#
# Requests Cori GPU nodes:
#SBATCH --constraint=gpu
#
# Each GPU node has 40 physical CPU cores
# (w/ 2 hyperthreads, so 80 virtual CPU cores)
# and 8 NVIDIA Tesla V100 GPUs.
#
# GPU: Assign 1 MPI tasks to each GPU
#SBATCH --tasks-per-node=8
#
# We want all 8 GPUs on the node
#SBATCH --gres=gpu:8
#
# Since we want the entire node (all the GPUs)
# with 8 MPI tasks, this is 80/8 = 10 virtual CPUs
# per MPI task.
#SBATCH --cpus-per-task=10
#
#################
# Queue & Job
#################
#
# Which queue to run in: debug, regular, premium, etc. ...
# For now, don't use --qos
## SBATCH --qos=debug
#
# Run for this much walltime: hh:mm:ss
#SBATCH --time=00:15:00
#
# Use this job name:
#SBATCH -J emu_gpu_test
#
# Send notification emails here:
#SBATCH --mail-user=eugene.willcox@gmail.com
#SBATCH --mail-type=ALL
#
# Which allocation to use:
#SBATCH -A m3018

# On the compute node, change to the directory we submitted from
cd $SLURM_SUBMIT_DIR

# OpenMP Configuration
# This configuration ignores the hyperthreads
# and assigns 1 OpenMP thread/physical core
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=5

srun --cpu_bind=cores ./main3d.gnu.DEBUG.TPROF.MPI.CUDA.ex inputs_bipolar_test
