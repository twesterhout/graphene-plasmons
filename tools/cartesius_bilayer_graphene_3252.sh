#!/usr/bin/env bash
#
#SBATCH -p gpu -t 12:00:00
#SBATCH -n 1 -c 16

set -e
set -o pipefail

export OMP_NUM_THREADS=8          # 2 processes with 8 cores each
kT=0.0256                         # Temperature in eV
mu=0                              # Chemical potential μ in eV
damping=0.001                     # Damping η in eV

declare -a freq                   # Comma-separated list of frequencies ω in eV
freq[0]=$(seq -s, 0.0 0.1 22.0)   # we split it into two parts since the node has
freq[1]=$(seq -s, 0.05 0.1 22.05) # 2 gpus

if [ $# -ne 2 ]; then
	echo "Usage: $0 <input_file> <output_file>"
	exit 1
fi
input_file=$1
output_file=$2

# Compute χ
for device in 0 1; do
	unbuffer julia -e 'import Plasmons; Plasmons.julia_main()' -- \
    	--kT "$kT" --mu "$mu" --damping "$damping" \
    	--hamiltonian H \
    	--frequency "${freq[$device]}" \
    	--cuda "$device" \
    	"$input_file" "$output_file.$device" \
		>slurm-${SLURM_JOB_ID}_$device.out 2>&1 &
done
wait
