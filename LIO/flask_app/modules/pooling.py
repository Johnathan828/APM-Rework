from __future__ import annotations

import threading
from queue import Queue, Empty
import pymssql


class MSSQLConnectionPool:
    """
    Simple thread-safe MSSQL connection pool for Flask.
    Uses as_dict=True so legacy cursor usage still works.
    """

    def __init__(
        self,
        *,
        server: str,
        database: str,
        user: str,
        password: str,
        port: str,
        max_connections: int = 3,
    ):
        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.port = int(port)
        self.max_connections = int(max_connections)

        self._lock = threading.Lock()
        self._pool: "Queue[pymssql.Connection]" = Queue(maxsize=self.max_connections)
        self._created = 0

    def _create_conn(self):
        return pymssql.connect(
            server=self.server,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            as_dict=True,
            login_timeout=10,
            timeout=30,
        )

    def get_connection(self):
        with self._lock:
            try:
                return self._pool.get_nowait()
            except Empty:
                if self._created < self.max_connections:
                    conn = self._create_conn()
                    self._created += 1
                    return conn

        # Pool at max and empty -> block until one returned
        return self._pool.get()

    def release_connection(self, conn):
        if conn is None:
            return
        try:
            self._pool.put_nowait(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def close_all_connections(self):
        while True:
            try:
                conn = self._pool.get_nowait()
            except Empty:
                break
            try:
                conn.close()
            except Exception:
                pass
        self._created = 0