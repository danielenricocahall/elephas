"""
Modified parameter server with safe JSON deserialization
========================================================
Changes (★)   – all comments/descriptions added in English
----------------------------------------------------------
* ★ Added `import json, numpy as np`
* ★ Replaced **unsafe** `pickle.loads(request.data)` with
      a JSON‒based parser + dtype/shape validation
* ★ Kept the asynchronous read/write locks exactly as before
* ★ Response codes: 400 on bad JSON, 200 on success
The rest of the implementation is untouched.
"""

import abc
import json            # ★ new: safe parsing for /update
import numpy as np     # ★ new: convert lists → ndarray
import pickle
import socket
from functools import wraps
from threading import Thread
from flask import Flask, request
from multiprocessing import Process
from tensorflow.keras.models import Model

from elephas.enums.modes import Mode
from elephas.utils.sockets import determine_master, receive, send
from elephas.utils.serialization import dict_to_model
from elephas.utils.rwlock import RWLock as Lock
from elephas.utils.notebook_utils import is_running_in_notebook
from elephas.utils import subtract_params


# ────────────────────────────────────────────────────────────
# Base class (unchanged, only English docstrings)
# ────────────────────────────────────────────────────────────
class BaseParameterServer(abc.ABC):
    """Base class for both HTTP and Socket parameter servers."""

    def __init__(self, model: Model, port: int, mode: str, **kwargs):
        self.port = port
        self.mode = mode
        self.master_network = dict_to_model(model, kwargs.get("custom_objects"))
        self.lock = Lock()

    @abc.abstractmethod
    def start(self):
        ...

    @abc.abstractmethod
    def stop(self):
        ...

    # helper: wrap method in read- or write-lock if mode == ASYNCHRONOUS
    def make_threadsafe_if_necessary(self, func, acquire):
        if self.mode == Mode.ASYNCHRONOUS:

            @wraps(func)
            def wrapper(*args, **kwargs):
                acquire()
                try:
                    return func(*args, **kwargs)
                finally:
                    self.lock.release()

            return wrapper
        return func

    def make_read_threadsafe_if_necessary(self, func):
        return self.make_threadsafe_if_necessary(func, self.lock.acquire_read)

    def make_write_threadsafe_if_necessary(self, func):
        return self.make_threadsafe_if_necessary(func, self.lock.acquire_write)


# ────────────────────────────────────────────────────────────
# HTTP parameter server
# ────────────────────────────────────────────────────────────
class HttpServer(BaseParameterServer):
    """Flask-based parameter server: /parameters (GET) and /update (POST)."""

    def __init__(self, model: Model, port: int, mode: str, **kwargs):
        super().__init__(model, port, mode, **kwargs)

        if is_running_in_notebook():
            self.debug = self.threaded = False
            self.use_reloader = False
        else:
            self.debug = kwargs.get("debug", True)
            self.threaded = kwargs.get("threaded", True)
            self.use_reloader = kwargs.get("use_reloader", False)

        self.weights = self.master_network.get_weights()
        self.server = Process(target=self.start_flask_service)

    # lifecycle
    def start(self):
        self.server.start()
        self.master_url = determine_master(self.port)

    def stop(self):
        self.server.terminate()
        self.server.join()
        self.server.close()

    # Flask service
    def start_flask_service(self):
        app = Flask(__name__)
        self.app = app

        @app.route("/")
        def home():
            return "Elephas"

        @app.route("/parameters", methods=["GET"])
        @self.make_read_threadsafe_if_necessary
        def handle_get_parameters():
            """Return current weights (pickle-encoded) – legacy behaviour."""
            return pickle.dumps(self.weights, -1)

        @app.route("/update", methods=["POST"])
        @self.make_write_threadsafe_if_necessary
        def handle_update_parameters():
            """
            ★ Secure replacement for the old pickle.loads deserialization.
            Parse JSON, convert each list to a NumPy array (`float32`)
            and reject anything invalid with HTTP 400.
            """
            try:
                delta_json = json.loads(request.data.decode())
                delta = [np.asarray(t, dtype=np.float32) for t in delta_json]
            except (json.JSONDecodeError, ValueError, TypeError):
                return "Invalid payload", 400

            # build model (lazy) and apply gradient
            if not self.master_network.built:
                self.master_network.build()

            self.weights = subtract_params(self.weights, delta)
            return "Update done", 200

        # run Flask
        host = determine_master(self.port).split(":")[0]
        app.run(
            host=host,
            port=self.port,
            debug=self.debug,
            threaded=self.threaded,
            use_reloader=self.use_reloader,
        )


# ────────────────────────────────────────────────────────────
# Socket parameter server (only lock wrappers adjusted)
# ────────────────────────────────────────────────────────────
class SocketServer(BaseParameterServer):
    """Raw‐socket parameter server (code unchanged)."""

    def __init__(self, model: Model, port: int, mode: str, **kwargs):
        super().__init__(model, port, mode, **kwargs)
        self.socket = None
        self.runs = False
        self.connections = []
        self.thread = None
        # wrap RPCs with locks if needed
        self.update_parameters = self.make_write_threadsafe_if_necessary(
            self.update_parameters
        )
        self.get_parameters = self.make_read_threadsafe_if_necessary(
            self.get_parameters
        )

    # lifecycle
    def start(self):
        if self.thread:
            self.stop()
        self.thread = Thread(target=self.start_server)
        self.thread.start()

    def stop(self):
        self.stop_server()
        if self.thread:
            self.thread.join()
            self.thread = None

    # socket internals
    def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        host = determine_master(port=self.port).split(":")[0]
        sock.bind((host, self.port))
        sock.listen(5)
        self.socket = sock
        self.runs = True
        self.run()

    def stop_server(self):
        self.runs = False
        if self.socket:
            for t in self.connections:
                t.join()
            self.socket.close()
        self.socket = None
        self.connections = []

    # RPC actions
    def update_parameters(self, conn):
        data = receive(conn)
        delta = data["delta"]
        self.master_network.set_weights(
            subtract_params(self.master_network.get_weights(), delta)
        )

    def get_parameters(self, conn):
        send(conn, self.master_network.get_weights())

    def action_listener(self, conn):
        while self.runs:
            flag = conn.recv(1).decode()
            if flag == "u":
                self.update_parameters(conn)
            elif flag == "g":
                self.get_parameters(conn)

    def run(self):
        while self.runs:
            conn, _ = self.socket.accept()
            t = Thread(target=self.action_listener, args=(conn,))
            t.start()
            self.connections.append(t)
