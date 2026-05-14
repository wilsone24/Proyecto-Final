"""
local_spark.py
--------------
Provides a local SparkSession with Delta Lake support and a minimal
dbutils mock so that the Databricks notebooks can be imported and
executed outside Databricks.

Usage in every test file:
    from tests.local_spark import spark, dbutils
"""

import os
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# JVM compatibility flags for Java 17/21
# Must be set BEFORE SparkContext is created.
# Fixes: "getSubject is not supported" (Hadoop/Java17 incompatibility)
# ---------------------------------------------------------------------------
_JVM_OPENS = " ".join([
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
])
os.environ["JAVA_TOOL_OPTIONS"] = _JVM_OPENS

# Suppress winutils.exe warning on Windows (Hadoop not needed for local runs)
os.environ.setdefault("HADOOP_HOME", os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYSPARK_PYTHON", "python")

def _build_session() -> SparkSession:
    try:
        from delta import configure_spark_with_delta_pip

        builder = (
            SparkSession.builder
            .appName("local-test")
            .master("local[*]")
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.driver.extraJavaOptions", _JVM_OPENS)
            .config("spark.executor.extraJavaOptions", _JVM_OPENS)
        )
        return configure_spark_with_delta_pip(builder).getOrCreate()

    except ImportError:
        # Fall back to plain Spark (no Delta) if delta-spark is not installed
        return (
            SparkSession.builder
            .appName("local-test")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )


spark = _build_session()
spark.sparkContext.setLogLevel("ERROR")   # suppress INFO/WARN noise

# ---------------------------------------------------------------------------
# dbutils mock
# ---------------------------------------------------------------------------
class _Widgets:
    """Mimics dbutils.widgets with simple in-memory storage."""

    def __init__(self):
        self._store: dict = {}

    def text(self, name: str, default: str) -> None:
        """Register a widget with its default value."""
        self._store[name] = default

    def get(self, name: str) -> str:
        if name not in self._store:
            raise KeyError(f"Widget '{name}' not found. Call dbutils.widgets.text() first.")
        return self._store[name]


class _Dbutils:
    widgets = _Widgets()


dbutils = _Dbutils()
