import socket

import pytest


@pytest.fixture
def unused_tcp_port():
    """Reserve a TCP port for the test's parameter server.

    SO_REUSEADDR is set BEFORE bind so when the server later rebinds the
    same port — possibly while the kernel still holds a TIME_WAIT entry from
    a prior test — the bind succeeds. The original implementation called
    setsockopt after bind, which has no effect.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]
