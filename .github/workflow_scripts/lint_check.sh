#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env

function lint_check {
    # black
    ruff format --diff "$1/"
    # isort
    ruff check --select I "$1/"
}

function lint_check_all {
    lint_check CI autogluon common core docs examples features multimodal tabular timeseries
}

bandit -r multimodal/src -ll --exclude "multimodal/src/autogluon/multimodal/configs/pretrain/*"
lint_check_all
ruff check timeseries/
