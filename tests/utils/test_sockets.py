import socket
import threading
import time

import pytest

from elephas.utils.sockets import wait_until_listening


def _reserve_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _open_listener(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def test_wait_until_listening_returns_when_server_ready():
    port = _reserve_port()
    sock = _open_listener(port)
    try:
        t0 = time.monotonic()
        wait_until_listening("127.0.0.1", port, timeout=2)
        assert time.monotonic() - t0 < 0.5
    finally:
        sock.close()


def test_wait_until_listening_blocks_until_server_comes_up():
    port = _reserve_port()
    delay_seconds = 0.4
    holder = {}

    def delayed_start():
        time.sleep(delay_seconds)
        holder["sock"] = _open_listener(port)

    thread = threading.Thread(target=delayed_start, daemon=True)
    thread.start()
    try:
        t0 = time.monotonic()
        wait_until_listening("127.0.0.1", port, timeout=3)
        elapsed = time.monotonic() - t0
        assert elapsed >= delay_seconds * 0.8
        assert elapsed < 2
    finally:
        thread.join(timeout=1)
        if "sock" in holder:
            holder["sock"].close()


def test_wait_until_listening_raises_timeout_when_server_never_starts():
    port = _reserve_port()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="did not start listening"):
        wait_until_listening("127.0.0.1", port, timeout=0.3)
    # Roughly honors the deadline (allow 1s slack for poll cadence + connect timeout).
    assert time.monotonic() - t0 < 1.5
