#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
target_dir=$(dirname -- ${script_dir})

. "${script_dir}/utils.sh"

(
    cd $target_dir
    cargo test --target-dir $target_dir --workspace
    check_operation_success $? "Testing"
    cargo check --target-dir $target_dir --workspace
    check_operation_success $? "Typechecking"
    cargo fmt --manifest-path "${target_dir}/Cargo.toml" --all
    check_operation_success $? "Formating"
    cargo clippy --workspace --tests --examples -- -D warnings
    check_operation_success $? "Linting"
)