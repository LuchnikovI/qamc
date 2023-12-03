#!/usr/bin/env bash
#SBATCH --job-name=qamc
#SBATCH --output=logs.txt
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Ilia.Luchnikov@tii.ae

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ci_dir="${script_dir}/../ci"
src_dir="${script_dir}/../src"

# std normal ensemble
. "${ci_dir}/runner.sh" "${src_dir}/run.py" parameters=large

# discrete ensemble
. "${ci_dir}/runner.sh" "${src_dir}/run.py" \
    parameters=large \
    couplings_ensemble=discrete \
    local_fields_ensemble=discrete