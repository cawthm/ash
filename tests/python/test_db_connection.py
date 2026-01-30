"""Tests for database connection utility."""

import tempfile
from pathlib import Path

import pytest

from data.exploration.db_connection import DatabaseConfig, DatabaseConnection, get_connection


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_from_env_loads_credentials(self) -> None:
        """Test loading database credentials from .env file."""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# Database Configuration\n")
            f.write("DB_HOST=localhost\n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME=testdb\n")
            f.write("DB_USER=testuser\n")
            f.write("DB_PASSWORD=testpass\n")
            env_path = Path(f.name)

        try:
            config = DatabaseConfig.from_env(env_path)

            assert config.host == "localhost"
            assert config.port == 5432
            assert config.database == "testdb"
            assert config.user == "testuser"
            assert config.password == "testpass"
        finally:
            env_path.unlink()

    def test_from_env_handles_whitespace(self) -> None:
        """Test that .env parser handles whitespace correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("  DB_HOST  =  localhost  \n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME = testdb\n")
            f.write("DB_USER=testuser\n")
            f.write("DB_PASSWORD=testpass\n")
            env_path = Path(f.name)

        try:
            config = DatabaseConfig.from_env(env_path)

            assert config.host == "localhost"
            assert config.database == "testdb"
        finally:
            env_path.unlink()

    def test_from_env_ignores_comments(self) -> None:
        """Test that .env parser ignores comment lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("DB_HOST=localhost\n")
            f.write("# Another comment\n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME=testdb\n")
            f.write("DB_USER=testuser\n")
            f.write("DB_PASSWORD=testpass\n")
            env_path = Path(f.name)

        try:
            config = DatabaseConfig.from_env(env_path)

            assert config.host == "localhost"
            assert config.port == 5432
        finally:
            env_path.unlink()

    def test_from_env_missing_file_raises_error(self) -> None:
        """Test that missing .env file raises FileNotFoundError."""
        nonexistent_path = Path("/tmp/nonexistent_file_12345.env")

        with pytest.raises(FileNotFoundError, match=".env file not found"):
            DatabaseConfig.from_env(nonexistent_path)

    def test_from_env_missing_variables_raises_error(self) -> None:
        """Test that missing required variables raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_PORT=5432\n")
            # Missing DB_NAME, DB_USER, DB_PASSWORD
            env_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required environment variables"):
                DatabaseConfig.from_env(env_path)
        finally:
            env_path.unlink()

    def test_from_env_partial_variables_raises_error(self) -> None:
        """Test that partially missing variables raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME=testdb\n")
            f.write("DB_USER=testuser\n")
            # Missing DB_PASSWORD
            env_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required environment variables"):
                DatabaseConfig.from_env(env_path)
        finally:
            env_path.unlink()


class TestDatabaseConnection:
    """Tests for DatabaseConnection.

    Note: These tests verify the interface but don't require a live database.
    Integration tests with actual database are in test_data_exploration.py.
    """

    def test_context_manager_requires_connection(self) -> None:
        """Test that query methods require connection to be established."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
        )
        conn = DatabaseConnection(config)

        # Should raise error when used outside context manager
        with pytest.raises(RuntimeError, match="Database connection not established"):
            conn.query("SELECT 1")

    def test_query_one_requires_connection(self) -> None:
        """Test that query_one requires connection to be established."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
        )
        conn = DatabaseConnection(config)

        with pytest.raises(RuntimeError, match="Database connection not established"):
            conn.query_one("SELECT 1")

    def test_execute_requires_connection(self) -> None:
        """Test that execute requires connection to be established."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
        )
        conn = DatabaseConnection(config)

        with pytest.raises(RuntimeError, match="Database connection not established"):
            conn.execute("UPDATE test SET value = 1")


class TestGetConnection:
    """Tests for get_connection convenience function."""

    def test_get_connection_creates_connection(self) -> None:
        """Test that get_connection creates a DatabaseConnection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME=testdb\n")
            f.write("DB_USER=testuser\n")
            f.write("DB_PASSWORD=testpass\n")
            env_path = Path(f.name)

        try:
            conn = get_connection(env_path)

            assert isinstance(conn, DatabaseConnection)
            assert conn.config.host == "localhost"
            assert conn.config.database == "testdb"
        finally:
            env_path.unlink()
