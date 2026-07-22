import os
import tempfile
from unittest import mock

import pytest

from autogluon.multimodal.utils.download import (
    _check_url_points_to_public_host,
    download,
    is_url,
)

# URLs whose host resolves to a non-public address. These must never be fetched.
NON_PUBLIC_URLS = [
    "http://127.0.0.1:9999/internal/secret",  # loopback (IPv4)
    "http://localhost:8080/admin",  # loopback (hostname)
    "http://169.254.169.254/latest/meta-data/",  # link-local / cloud metadata endpoint
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",  # metadata credentials
    "http://10.0.0.1/admin",  # private (10.0.0.0/8)
    "http://192.168.1.1/api",  # private (192.168.0.0/16)
    "http://172.16.0.1/api",  # private (172.16.0.0/12)
    "http://[::1]:8000/",  # loopback (IPv6)
    "http://0.0.0.0/",  # unspecified
    "http://224.0.0.1/",  # multicast
]


@pytest.mark.parametrize("url", NON_PUBLIC_URLS)
def test_check_url_points_to_public_host_rejects_non_public(url):
    with pytest.raises(ValueError):
        _check_url_points_to_public_host(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9999/internal/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/admin",
    ],
)
def test_download_rejects_non_public_host(url):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "result.bin")
        with pytest.raises(ValueError):
            download(url, path=dest, retries=0)
        # Nothing should have been written to disk.
        assert not os.path.exists(dest)


def test_download_does_not_request_non_public_host():
    """A rejected URL must be blocked before any HTTP request is issued."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "result.bin")
        with mock.patch("autogluon.multimodal.utils.download.requests.get") as mock_get:
            with pytest.raises(ValueError):
                download("http://169.254.169.254/latest/meta-data/", path=dest, retries=3)
        mock_get.assert_not_called()


def test_check_url_points_to_public_host_allows_public():
    # A well-known public host should pass. getaddrinfo is patched so the test does not
    # depend on live DNS.
    with mock.patch(
        "autogluon.multimodal.utils.download.socket.getaddrinfo",
        return_value=[(0, 0, 0, "", ("93.184.216.34", 0))],
    ):
        _check_url_points_to_public_host("http://example.com/dataset.zip")


def test_check_url_points_to_public_host_rejects_dns_resolving_to_private():
    # Even a normal-looking hostname must be rejected if it resolves to a private address.
    with mock.patch(
        "autogluon.multimodal.utils.download.socket.getaddrinfo",
        return_value=[(0, 0, 0, "", ("10.1.2.3", 0))],
    ):
        with pytest.raises(ValueError):
            _check_url_points_to_public_host("http://sneaky.example.com/dataset.zip")


def test_check_url_points_to_public_host_rejects_unresolvable():
    import socket as _socket

    with mock.patch(
        "autogluon.multimodal.utils.download.socket.getaddrinfo",
        side_effect=_socket.gaierror("name resolution failed"),
    ):
        with pytest.raises(ValueError):
            _check_url_points_to_public_host("http://does-not-resolve.invalid/x.zip")


def test_check_url_points_to_public_host_rejects_missing_host():
    with pytest.raises(ValueError):
        _check_url_points_to_public_host("not-a-url")


def test_is_url_still_recognizes_url_shapes():
    # is_url is unchanged: it only checks URL shape, not host reachability.
    assert is_url("http://example.com/data.zip") is True
    assert is_url("http://127.0.0.1:9999/internal/secret") is True
    assert is_url("/local/path/to/file.zip") is False
    assert is_url(1234) is False
