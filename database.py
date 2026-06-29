"""
database.py — PostgreSQL integration for the Quality Control System.

Uses a ThreadedConnectionPool so the Streamlit thread and background
operations can share connections safely.
"""
import logging
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.pool
import psycopg2.extras

from config import DB_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DDL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS production_log (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    class_name  VARCHAR(50)  NOT NULL,
    track_id    INTEGER      NOT NULL,
    session_id  VARCHAR(64)  NOT NULL,

    -- Guarantee that the same physical object (track_id) inside
    -- the same recording session is stored at most once.
    CONSTRAINT uq_track_session UNIQUE (track_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_pl_timestamp  ON production_log (timestamp  DESC);
CREATE INDEX IF NOT EXISTS idx_pl_class      ON production_log (class_name);
CREATE INDEX IF NOT EXISTS idx_pl_session    ON production_log (session_id);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """Thread-safe PostgreSQL helper.

    Designed to be used as a ``st.cache_resource`` singleton.
    """

    def __init__(self) -> None:
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._connected = False
        self._error: Optional[str] = None
        self._init()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                **DB_CONFIG,
            )
            self._create_schema()
            self._connected = True
            logger.info("PostgreSQL connection pool ready.")
        except Exception as exc:
            self._error = str(exc)
            logger.warning("PostgreSQL unavailable — running in offline mode. Error: %s", repr(exc))

    def _create_schema(self) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()

    # ── helpers ────────────────────────────────────────────────────────────

    def _get(self):
        return self._pool.getconn()

    def _put(self, conn) -> None:
        self._pool.putconn(conn)

    # ── write ──────────────────────────────────────────────────────────────

    def log_detection(
        self,
        class_name: str,
        track_id: int,
        session_id: str,
    ) -> bool:
        """Insert a crossing event.

        Returns ``True`` if a new row was inserted, ``False`` on duplicate
        or database error.
        """
        if not self._connected:
            return False

        sql = """
        INSERT INTO production_log (timestamp, class_name, track_id, session_id)
        VALUES (NOW(), %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_track_session DO NOTHING
        RETURNING id;
        """
        conn = self._get()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (class_name, track_id, session_id))
                inserted = cur.fetchone() is not None
            conn.commit()
            return inserted
        except Exception as exc:
            conn.rollback()
            logger.error("log_detection failed: %s", exc)
            return False
        finally:
            self._put(conn)

    def get_stats_by_date_range(self, start_date, end_date) -> list[dict]:
        if not self._connected: return []
        sql = """
            SELECT date_trunc('hour', timestamp) AS hour, class_name, COUNT(*)::INTEGER AS count
            FROM production_log WHERE timestamp::date BETWEEN %s AND %s
            GROUP BY 1, 2 ORDER BY 1 ASC, 2;
        """
        conn = self._get()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (start_date, end_date))
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("get_stats_by_date_range failed: %s", exc)
            return []
        finally:
            self._put(conn)

    # ── read ───────────────────────────────────────────────────────────────

    def get_hourly_stats(self, hours: int = 24) -> list[dict]:
        """Return counts grouped by (hour, class_name) for the last *hours* hours."""
        if not self._connected:
            return []

        sql = """
        SELECT
            date_trunc('hour', timestamp)   AS hour,
            class_name,
            COUNT(*)::INTEGER               AS count
        FROM  production_log
        WHERE timestamp > NOW() - (%s || ' hours')::INTERVAL
        GROUP BY 1, 2
        ORDER BY 1 ASC, 2;
        """
        conn = self._get()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (str(hours),))
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("get_hourly_stats failed: %s", exc)
            return []
        finally:
            self._put(conn)

    def get_session_counts(self, session_id: str) -> dict[str, int]:
        """Return per-class totals for a single session."""
        if not self._connected:
            return {}

        sql = """
        SELECT class_name, COUNT(*)::INTEGER AS cnt
        FROM   production_log
        WHERE  session_id = %s
        GROUP  BY class_name;
        """
        conn = self._get()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                return {row[0]: row[1] for row in cur.fetchall()}
        except Exception as exc:
            logger.error("get_session_counts failed: %s", exc)
            return {}
        finally:
            self._put(conn)

    def get_all_time_counts(self) -> dict[str, int]:
        """Return per-class grand totals across all sessions."""
        if not self._connected:
            return {}

        sql = """
        SELECT class_name, COUNT(*)::INTEGER AS cnt
        FROM   production_log
        GROUP  BY class_name;
        """
        conn = self._get()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                return {row[0]: row[1] for row in cur.fetchall()}
        except Exception as exc:
            logger.error("get_all_time_counts failed: %s", exc)
            return {}
        finally:
            self._put(conn)

    def get_recent_events(self, limit: int = 20) -> list[dict]:
        """Return the *limit* most recent log entries."""
        if not self._connected:
            return []

        sql = """
        SELECT id, timestamp, class_name, track_id, session_id
        FROM   production_log
        ORDER  BY timestamp DESC
        LIMIT  %s;
        """
        conn = self._get()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("get_recent_events failed: %s", exc)
            return []
        finally:
            self._put(conn)
