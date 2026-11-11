from unittest.mock import Mock

import boto3
import pytest

from autogluon.common.utils.s3_utils import delete_s3_prefix


@pytest.mark.parametrize("num_objects", [0, 1, 999, 1000, 1001, 2501])
def test_delete_s3_prefix_removes_all_pages(monkeypatch, num_objects):
    keys = [f"data/part-{i:05d}.parquet" for i in range(num_objects)]
    pages = [
        {"Contents": [{"Key": key} for key in keys[start : start + 1000]]} for start in range(0, num_objects, 1000)
    ] or [{}]
    resource = Mock()
    client = resource.meta.client
    client.list_objects.return_value = pages[0]
    client.get_paginator.return_value.paginate.return_value = iter(pages)
    client.delete_objects.return_value = {}
    monkeypatch.setattr(boto3, "resource", lambda service: resource)

    delete_s3_prefix("bucket", "data/")

    deleted = [obj["Key"] for call in client.delete_objects.call_args_list for obj in call.kwargs["Delete"]["Objects"]]
    assert deleted == keys
    assert all(call.kwargs["Bucket"] == "bucket" for call in client.delete_objects.call_args_list)
    assert all(0 < len(call.kwargs["Delete"]["Objects"]) <= 1000 for call in client.delete_objects.call_args_list)
    pagination_call = client.get_paginator.return_value.paginate.call_args
    request = pagination_call.kwargs if pagination_call is not None else client.list_objects.call_args.kwargs
    assert request["Bucket"] == "bucket"
    assert request["Prefix"] == "data/"
    if num_objects == 0:
        client.delete_objects.assert_not_called()


def test_delete_s3_prefix_reports_partial_delete_failures(monkeypatch):
    resource = Mock()
    client = resource.meta.client
    pages = [{"Contents": [{"Key": "data/blocked.parquet"}]}]
    client.list_objects.return_value = pages[0]
    client.get_paginator.return_value.paginate.return_value = iter(pages)
    client.delete_objects.return_value = {"Errors": [{"Key": "data/blocked.parquet", "Code": "AccessDenied"}]}
    monkeypatch.setattr(boto3, "resource", lambda service: resource)

    with pytest.raises(RuntimeError, match="AccessDenied"):
        delete_s3_prefix("bucket", "data/")
