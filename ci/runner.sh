#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

. "${script_dir}/ensure_image.sh"

singularity exec --cleanenv "${QAMC_IMAGE_NAME}.sif" "$@"