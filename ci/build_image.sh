#!/usr/bin/env bash

ci_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${ci_dir}/utils.sh"

project_dir="$(dirname ${ci_dir})"

# ------------------------------------------------------------------------------------------

log INFO "Building an image..."
cat > "${QAMC_IMAGE_NAME}.def" <<EOF

Bootstrap: docker
From: ${QAMC_RUST_IMAGE}
Stage: build
%files
    "${project_dir}/src" /qamc/src
    "${project_dir}/Cargo.toml" /qamc/Cargo.toml
    "${project_dir}/Cargo.lock" /qamc/Cargo.lock
    "${project_dir}/ci" /qamc/ci
%post
    rustup component add rustfmt
    rustup component add clippy
    /qamc/ci/check_rust.sh
    apt-get update && apt-get install -y python3-pip python3-venv
    python3 -m pip install --break-system-packages --no-cache --upgrade \
        pip \
        setuptools \
        maturin \
        patchelf \
        numpy
    python3 -m maturin build --release --manifest-path /qamc/Cargo.toml

Bootstrap: docker
From: ${QAMC_BASE_IMAGE}
Stage: final
%files from build
    /qamc/target/wheels /qamc/wheels
%post
    apt-get update && apt-get install -y python3-pip
    python3 -m pip install --break-system-packages --no-cache-dir --upgrade \
        pip \
        setuptools \
        numpy==1.25.1 \
        scipy==1.11.1 \
        matplotlib==3.7.2 \
        pytest==7.4.0 \
        pyyaml==6.0 \
        mypy==1.4.1 \
        h5py==3.9.0
    for wheel in /qamc/wheels/*
    do
        python3 -m pip install --break-system-packages "\${wheel}"
    done

EOF

# ------------------------------------------------------------------------------------------

if singularity build -F "${QAMC_IMAGE_NAME}.sif" "${QAMC_IMAGE_NAME}.def";
then
    log INFO "Base image ${QAMC_IMAGE_NAME} has been built"
    rm -f "${QAMC_IMAGE_NAME}.def"
else
    log ERROR "Failed to build a base image ${QAMC_IMAGE_NAME}"
    rm -f "${QAMC_IMAGE_NAME}.def"
    exit 1
fi