"""Schema exploration script for options_data and stock_trades tables.

Queries the database to discover table schemas, column types, and basic statistics.
"""

from typing import Any

from data.exploration.db_connection import get_connection


def get_table_schema(table_name: str) -> list[dict[str, Any]]:
    """Get schema information for a table.

    Args:
        table_name: Name of the table to query.

    Returns:
        List of column information dictionaries with keys:
        - column_name: Column name
        - data_type: PostgreSQL data type
        - is_nullable: 'YES' or 'NO'
        - column_default: Default value (if any)
    """
    with get_connection() as db:
        schema_query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """
        return db.query(schema_query, (table_name,))


def get_table_row_count(table_name: str) -> int:
    """Get the approximate number of rows in a table (fast).

    Uses PostgreSQL statistics for fast approximate counts on large tables.

    Args:
        table_name: Name of the table to query.

    Returns:
        Approximate number of rows in the table.
    """
    with get_connection() as db:
        # Use pg_class statistics for fast approximate count
        result = db.query_one(
            """
            SELECT reltuples::bigint AS count
            FROM pg_class
            WHERE relname = %s
            """,
            (table_name,),
        )
        return int(result["count"]) if result and result["count"] else 0


def explore_options_data() -> dict[str, Any]:
    """Explore the options_data table schema and basic stats.

    Returns:
        Dictionary with schema, row count, and sample data.
    """
    print("Exploring options_data table...")

    schema = get_table_schema("options_data")
    row_count = get_table_row_count("options_data")

    # Get sample row
    with get_connection() as db:
        sample = db.query_one("SELECT * FROM options_data LIMIT 1")

    # Get time range
    with get_connection() as db:
        time_range = db.query_one("""
            SELECT
                MIN(time) as min_time,
                MAX(time) as max_time
            FROM options_data
        """)

    # Get unique symbols (limit to first 100 for performance)
    with get_connection() as db:
        symbols = db.query(
            """SELECT DISTINCT underlying_ticker FROM options_data
               ORDER BY underlying_ticker LIMIT 100"""
        )

    return {
        "table_name": "options_data",
        "schema": schema,
        "row_count": row_count,
        "sample_row": sample,
        "time_range": time_range,
        "unique_symbols": [s["underlying_ticker"] for s in symbols],
        "symbol_count": len(symbols),
    }


def explore_stock_trades() -> dict[str, Any]:
    """Explore the stock_trades table schema and basic stats.

    Returns:
        Dictionary with schema, row count, and sample data.
    """
    print("Exploring stock_trades table...")

    schema = get_table_schema("stock_trades")
    row_count = get_table_row_count("stock_trades")

    # Get sample row
    with get_connection() as db:
        sample = db.query_one("SELECT * FROM stock_trades LIMIT 1")

    # Get time range
    with get_connection() as db:
        time_range = db.query_one("""
            SELECT
                MIN(time) as min_time,
                MAX(time) as max_time
            FROM stock_trades
        """)

    # Get unique symbols (limit to first 100 for performance)
    with get_connection() as db:
        symbols = db.query(
            "SELECT DISTINCT ticker FROM stock_trades ORDER BY ticker LIMIT 100"
        )

    return {
        "table_name": "stock_trades",
        "schema": schema,
        "row_count": row_count,
        "sample_row": sample,
        "time_range": time_range,
        "unique_symbols": [s["ticker"] for s in symbols],
        "symbol_count": len(symbols),
    }


def print_exploration_results(results: dict[str, Any]) -> None:
    """Pretty-print exploration results.

    Args:
        results: Dictionary from explore_options_data() or explore_stock_trades().
    """
    print(f"\n{'='*80}")
    print(f"Table: {results['table_name']}")
    print(f"{'='*80}")

    print(f"\nRow Count (approx): {results['row_count']:,}")

    print("\nTime Range:")
    if results["time_range"]:
        print(f"  Min: {results['time_range']['min_time']}")
        print(f"  Max: {results['time_range']['max_time']}")

    print(f"\nSymbols: {results['symbol_count']} unique (showing up to 100)")
    print(f"  {', '.join(results['unique_symbols'][:10])}")
    if results["symbol_count"] > 10:
        print(f"  ... and {results['symbol_count'] - 10} more (up to 100 total shown)")

    print(f"\nSchema ({len(results['schema'])} columns):")
    for col in results["schema"]:
        nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
        default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
        print(f"  {col['column_name']:<30} {col['data_type']:<20} {nullable}{default}")

    if results["sample_row"]:
        print("\nSample Row:")
        for key, value in results["sample_row"].items():
            print(f"  {key:<30} {value}")


if __name__ == "__main__":
    # Explore both tables
    try:
        options_results = explore_options_data()
        print_exploration_results(options_results)

        trades_results = explore_stock_trades()
        print_exploration_results(trades_results)

        print(f"\n{'='*80}")
        print("Exploration complete!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\nError during exploration: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print("  2. Database credentials in .env are correct")
        print("  3. Tables 'options_data' and 'stock_trades' exist")
        raise
