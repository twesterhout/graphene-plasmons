#!/usr/bin/env bash
#
#SBATCH -p gpu -t 5-00:00:00
#SBATCH -n 1 -c 16

set -e
set -o pipefail

export OMP_NUM_THREADS=8          # 2 processes with 8 cores each
kT=0.0256                         # Temperature in eV
rpa_mu=0                              # RPA chemical potential μ in eV
damping=0.001                     # Damping η in eV
k=10 # Sample size
hamiltonian_mu=1.34  # Hamiltonian chemical potential
theta=2 # Twisting angle
input_prefix="paper/input"
output_prefix="paper/output"
desired_frequencies=$(seq 0.75 0.0005 0.9999)
number_nodes=10

number_freq=$(echo "$desired_frequencies" | wc -l)
chunk_size=$(( (number_freq + (number_nodes - 1)) / number_nodes ))
echo "Splitting $number_freq frequencies into chunks of $chunk_size ..."

if [ $# -ne 1 ]; then
	echo "Usage: $0 <job_index>"
	exit 1
fi
job_index=$1
first_index=$(( 1 + job_index * chunk_size ))
local_freq=$(echo "$desired_frequencies" | tail -n +${first_index} | head -n ${chunk_size})

declare -a freq
for device in 0 1; do
	# Trick with temp is to avoid having a trailing comma
	temp=$(echo "$local_freq" | awk "NR % 2 == $device") 
	freq[$device]=$(echo -n "$temp" | tr '\n' ',')
done

for device in 0 1; do
	echo "Device $device will compute [${freq[$device]}] ..."
done

rm -f slurm-${SLURM_JOB_ID}_$device.out
echo "Computing polarizability ..."
for device in 0 1; do
	input_file="${input_prefix}/bilayer_graphene_k=${k}_μ=${hamiltonian_mu}_θ=${theta}.h5"
	output_file="${output_prefix}/output_k=${k}_μ=${hamiltonian_mu}_θ=${theta}_part=${job_index}.h5.${device}"
	unbuffer julia --project=/home/twest/graphene-plasmons -e 'import Plasmons; Plasmons.julia_main()' -- \
    	--kT "$kT" --mu "$rpa_mu" --damping "$damping" \
    	--hamiltonian H \
    	--frequency "${freq[$device]}" \
    	--cuda "$device" \
    	"$input_file" "$output_file" \
		&>>slurm-${SLURM_JOB_ID}_$device.out &
done
wait

echo "Computing EELS ..."
for device in 0 1; do
	input_file="${output_prefix}/output_k=${k}_μ=${hamiltonian_mu}_θ=${theta}_part=${job_index}.h5.${device}"
	output_file="${output_prefix}/loss_k=${k}_μ=${hamiltonian_mu}_θ=${theta}_part=${job_index}.h5.${device}"
	unbuffer julia --project=/home/twest/graphene-plasmons -e "
		using GraphenePlasmons
		if (GraphenePlasmons.hasmagma())
			@info \"Initializing MAGMA...\"
			GraphenePlasmons.magma_init()
			GraphenePlasmons.magma_setdevice(${device})
		end
		GraphenePlasmons.compute_leading_eigenvalues(\"${input_file}\", output = \"${output_file}\", n = 10)
		if (GraphenePlasmons.hasmagma())
			GraphenePlasmons.magma_finalize()
		end
	" &>>slurm-${SLURM_JOB_ID}_$device.out &
done
wait
