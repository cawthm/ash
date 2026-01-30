"""Database connection utility for data exploration.

Provides direct PostgreSQL connectivity to query options_data and stock_trades tables.
Uses credentials from .env file instead of MCP.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import RealDictCursor


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "DatabaseConfig":
        """Load database configuration from .env file.

        Args:
            env_path: Path to .env file. If None, looks in project root.

        Returns:
            DatabaseConfig instance with loaded credentials.

        Raises:
            FileNotFoundError: If .env file not found.
            ValueError: If required environment variables missing.
        """
        if env_path is None:
            # Default to project root
            env_path = Path(__file__).parent.parent.parent / ".env"

        if not env_path.exists():
            raise FileNotFoundError(f".env file not found at {env_path}")

        # Simple .env parser (avoiding dependency on python-dotenv)
        env_vars: dict[str, str] = {}
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        # Extract required variables
        required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [k for k in required if k not in env_vars]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

        return cls(
            host=env_vars["DB_HOST"],
            port=int(env_vars["DB_PORT"]),
            database=env_vars["DB_NAME"],
            user=env_vars["DB_USER"],
            password=env_vars["DB_PASSWORD"],
        )


class DatabaseConnection:
    """Context manager for PostgreSQL database connections."""

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize database connection.

        Args:
            config: Database configuration.
        """
        self.config = config
        self._conn: Connection | None = None

    def __enter__(self) -> "DatabaseConnection":
        """Open database connection."""
        self._conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def query(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        Args:
            sql: SQL query string.
            params: Query parameters (optional).

        Returns:
            List of rows as dictionaries.

        Raises:
            RuntimeError: If connection not established.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not established. Use context manager.")

        with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def query_one(self, sql: str, params: tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        """Execute a query and return first result as dict.

        Args:
            sql: SQL query string.
            params: Query parameters (optional).

        Returns:
            First row as dictionary, or None if no results.

        Raises:
            RuntimeError: If connection not established.
        """
        results = self.query(sql, params)
        return results[0] if results else None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> int:
        """Execute a statement (INSERT/UPDATE/DELETE) and return row count.

        Args:
            sql: SQL statement.
            params: Statement parameters (optional).

        Returns:
            Number of affected rows.

        Raises:
            RuntimeError: If connection not established.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not established. Use context manager.")

        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            self._conn.commit()
            return cursor.rowcount if cursor.rowcount is not None else 0


def get_connection(env_path: Path | None = None) -> DatabaseConnection:
    """Create a database connection from .env configuration.

    Args:
        env_path: Path to .env file. If None, looks in project root.

    Returns:
        DatabaseConnection instance (use as context manager).

    Example:
        >>> with get_connection() as db:
        ...     results = db.query("SELECT * FROM stock_trades LIMIT 10")
    """
    config = DatabaseConfig.from_env(env_path)
    return DatabaseConnection(config)
