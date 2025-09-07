import struct
from socket import gethostbyname, gethostname
import os


def determine_master(port: int = 4000) -> str:
    """Determine address of master so that workers
    can connect to it. If the environment variable
    SPARK_LOCAL_IP is set, that address will be used.

    :param port: port on which the application runs
    :return: Master address

    Example usage:
        SPARK_LOCAL_IP=127.0.0.1 spark-submit --master \
            local[8] examples/mllib_mlp.py
    """
    if os.environ.get("SPARK_LOCAL_IP"):
        return os.environ["SPARK_LOCAL_IP"] + ":" + str(port)
    else:
        return gethostbyname(gethostname()) + ":" + str(port)


def _receive_all(socket, num_bytes):
    """Reads `num_bytes` bytes from the specified socket.

    :param socket: open socket instance
    :param num_bytes: number of bytes to read

    :return: received data
    """

    buffer = b""
    buffer_size = 0
    bytes_left = num_bytes
    while buffer_size < num_bytes:
        data = socket.recv(bytes_left)
        delta = len(data)
        buffer_size += delta
        bytes_left -= delta
        buffer += data
    return buffer


def send_bytes(sock, b: bytes):
    sock.sendall(struct.pack("!I", len(b)))
    sock.sendall(b)


def recv_bytes(sock) -> bytes:
    import socket as _s

    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise _s.error("socket closed")
        hdr += chunk
    (n,) = struct.unpack("!I", hdr)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise _s.error("socket closed during payload")
        buf.extend(chunk)
    return bytes(buf)
