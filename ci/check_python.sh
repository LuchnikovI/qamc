#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
target_dir="$(dirname -- ${script_dir})/src"

. "${script_dir}/utils.sh"

. "${script_dir}/ensure_image.sh"

. "${script_dir}/runner.sh" black $target_dir
check_operation_success $? "Formatting"

. "${script_dir}/runner.sh" pytest $target_dir
check_operation_success $? "Testing"

. "${script_dir}/runner.sh" mypy $target_dir
check_operation_success $? "Type checking"

. "${script_dir}/runner.sh" pylint --fail-under=9 $target_dir
check_operation_success $? "Linting"