"""Database utilities for connecting to PostgreSQL and querying data."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def load_db_config() -> dict[str, str]:
    """
    Load database configuration from .env file.

    Returns:
        Dictionary with database connection parameters.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load .env from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if not env_path.exists():
        raise ValueError(
            f".env file not found at {env_path}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    load_dotenv(env_path)

    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    config = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Missing required environment variable: {var}")
        config[var] = value

    return config


def get_db_engine() -> Engine:
    """
    Create SQLAlchemy engine for database connection.

    Returns:
        SQLAlchemy Engine instance.
    """
    config = load_db_config()
    connection_string = (
        f"postgresql://{config['DB_USER']}:{config['DB_PASSWORD']}"
        f"@{config['DB_HOST']}:{config['DB_PORT']}/{config['DB_NAME']}"
    )
    return create_engine(connection_string)


def query_to_dataframe(query: str, **params: Any) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.

    Args:
        query: SQL query string (use :param for parameters).
        **params: Query parameters for safe parameter substitution.

    Returns:
        DataFrame with query results.

    Example:
        >>> df = query_to_dataframe(
        ...     "SELECT * FROM stock_trades WHERE symbol = :symbol LIMIT :n",
        ...     symbol='AAPL',
        ...     n=100
        ... )
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def get_pgpassword_env() -> str:
    """
    Get PGPASSWORD environment variable value for psql commands.

    Returns:
        Value to use for PGPASSWORD environment variable.

    Example:
        >>> import subprocess
        >>> env = os.environ.copy()
        >>> env['PGPASSWORD'] = get_pgpassword_env()
        >>> subprocess.run(['psql', '-U', 'postgres', '-c', 'SELECT 1'], env=env)
    """
    config = load_db_config()
    return config["DB_PASSWORD"]
