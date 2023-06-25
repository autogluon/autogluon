#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env

function lint_check {
    black --check --diff "$1/" --line-length "$2"
    isort --check --diff "$1/"
}

function lint_check_all {
    lint_check multimodal 119
    lint_check timeseries 119
    lint_check common 160
    lint_check core 160
    lint_check features 160
    lint_check tabular 160
}

bandit -r multimodal/src -ll
lint_check_all
