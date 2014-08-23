#!/bin/sh

#PBS -N Polyworld
#PBS -o Polyworld.out
#PBS -e Polyworld.err

#PBS -l nodes=3:ppn=6
#PBS -q m1060

cd /home/dougher1/poly/polyworld-comm

source /share/apps/Modules/3.2.10/init/sh
module load intel/13.1.0
module load gcc/4.9.1
module load cuda/6.0
module load gsl/1.16
module load python/2.7.5

time mpirun -prepend-rank -machinefile $PBS_NODEFILE ./Polyworld sheets.wf &> mpiout
