#!/usr/bin/env bash
#
#SBATCH -p gpu -t 5-00:00:00
#SBATCH -n 1 -c 16

set -e
set -o pipefail

module load magma/2.5.4-fosscuda-2020a

export OMP_NUM_THREADS=8          # 2 processes with 8 cores each

if [ $# -ne 1 ]; then
	echo "Usage: $0 <job_index>"
	exit 1
fi
job_index=$1
chunk_size=1
first_index=$((1 + job_index * chunk_size))

coulomb_theta=5

for device in 0 1; do
    rm -f slurm-${SLURM_JOB_ID}_$device.out
done
for f in $(find paper/output/ -name 'output_k=10_μ=1.34_θ=0_part=*.h5.0' | grep -E 'part=[0-7]\.' | sort | tail -n +${first_index} | head -n ${chunk_size}); do
	base_name="${f%.0}"
	out_name="${base_name/output_k/loss_k}"
	out_name="${out_name/θ=/α=${coulomb_theta}_θ=}"
	# echo "${out_name}"
    for device in 0 1; do
		unbuffer julia --project=/home/twest/graphene-plasmons -e "
			using GraphenePlasmons
			if (GraphenePlasmons.hasmagma())
				@info \"Initializing MAGMA ...\"
			    GraphenePlasmons.magma_init()
			    GraphenePlasmons.magma_setdevice(${device})
			end
			GraphenePlasmons.compute_leading_eigenvalues(\"${base_name}.${device}\", output = \"${out_name}.${device}\",
				 θ = ${coulomb_theta}, n = 10)
			if (GraphenePlasmons.hasmagma())
			    GraphenePlasmons.magma_finalize()
			end
		" &>>slurm-${SLURM_JOB_ID}_$device.out &
    done
    wait
done
