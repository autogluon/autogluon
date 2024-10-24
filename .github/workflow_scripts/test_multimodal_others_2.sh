#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/test_multimodal.sh

test_multimodal others_2/test_backward_compatibility.py "$ADDITIONAL_TEST_ARGS"
