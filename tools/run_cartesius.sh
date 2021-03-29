#!/usr/bin/env bash
#
#SBATCH -p gpu -t 12:00:00
#SBATCH -n 1 -c 16

set -e
set -o pipefail

export OMP_NUM_THREADS=8          # 2 processes with 8 cores each
t=2.7                             # Hopping parameter in eV
kT=0.0256                         # Temperature in eV
mu=0                              # Chemical potential μ in eV
damping=0.001                     # Damping η in eV

declare -a freq                   # Comma-separated list of frequencies ω in eV
freq[0]=$(seq -s, 0.0 0.1 18.0)   # we split it into two parts since the node has
freq[1]=$(seq -s, 0.05 0.1 18.05) # 2 gpus

# Generate input
if [ ! -e data/single_layer/input_1626.h5 ]; then
	julia --project=. -e 'using GraphenePlasmons;
		single_layer_graphene_1626("data/single_layer/input_1626.h5")'
fi

# Compute χ
for device in 0 1; do
	unbuffer julia -e 'import Plasmons; Plasmons.julia_main()' -- \
    	--kT "$kT" --mu "$mu" --damping "$damping" \
    	--hamiltonian H \
    	--frequency "${freq[$device]}" \
    	--cuda "$device" \
    	"data/single_layer/input_1626.h5" \
		"data/single_layer/polarizability_1626_${device}.h5" \
		>slurm-${SLURM_JOB_ID}_$device.out 2>&1 &
done
wait
