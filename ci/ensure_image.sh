#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# ------------------------------------------------------------------------------------------

if [[ -f "${QAMC_IMAGE_NAME}.sif" ]]; then
    :
else
    log INFO "${QAMC_IMAGE_NAME}.sif image has not been found"
    . "${script_dir}/build_image.sh"
fi