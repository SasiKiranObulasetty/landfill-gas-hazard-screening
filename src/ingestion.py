"""
ingestion.py - Automated external environmental and methane data ingestion pipeline.
AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills.

Provides:
- Modular source adapter architecture for public environmental observations.
- Verified connectors for EPA Envirofacts, Open-Meteo ambient landfill weather,
  Carbon Mapper satellite/airborne plume catalogs, and synthetic mock fixtures.
- Strict scientific unit and dimensional validation: ensures incompatible satellite
  plume flux (kg/h) or ambient weather are NEVER fed into the 3-gas downhole classifier.
- SQLite persistent database storage (data/external_monitoring.db) completely segregated
  from the historical Bridgeton research dataset (18,469 records in Excel).
- Ingestion telemetry: audit logs, deduplication, retry handling, and rate-limit compliance.
"""

import os
import sys
import re
import json
import time
import sqlite3
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse
import pandas as pd
import requests

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "external_monitoring.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion_pipeline")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class NormalizedObservation:
    """Standardized representation of an external environmental observation."""
    source_name: str
    source_record_id: str
    observation_timestamp: str  # ISO-8601 UTC string
    retrieval_timestamp: str    # ISO-8601 UTC string
    observation_type: str       # 'facility_regulatory', 'ambient_meteorology', 'satellite_plume', 'external_wellhead'
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    raw_payload_json: str = "{}"
    ch4_value: Optional[float] = None
    ch4_unit: Optional[str] = None
    co2_value: Optional[float] = None
    co2_unit: Optional[str] = None
    o2_value: Optional[float] = None
    o2_unit: Optional[str] = None
    auxiliary_metrics_json: str = "{}"
    is_classifier_compatible: bool = False
    incompatibility_reason: Optional[str] = None
    data_quality_status: str = "VERIFIED_PUBLIC"  # 'VERIFIED_PUBLIC', 'UNVERIFIED', 'TEST_FIXTURE'
    is_synthetic: bool = False


@dataclass
class IngestionSummary:
    """Summary record of an ingestion run."""
    run_id: str
    source_name: str
    start_time: str
    end_time: str
    status: str  # 'SUCCESS', 'FAILED', 'PARTIAL'
    records_retrieved: int = 0
    records_stored: int = 0
    records_duplicate: int = 0
    records_rejected: int = 0
    error_message: Optional[str] = None


# ==============================================================================
# DATABASE MANAGER
# ==============================================================================

class DatabaseManager:
    """
    Manages persistent storage for external observations and audit logs.
    Supports:
    1. Managed cloud PostgreSQL services (e.g. Supabase, Neon, AWS RDS) via PERSISTENT_DATABASE_URL / DATABASE_URL.
    2. Local SQLite fallback (data/external_monitoring.db) when cloud persistence is not configured.
    Ensures safe initialization, migrations, deduplication, and complete segregation
    from the historical Bridgeton research Excel file.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        connection_url: Optional[str] = None
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.raw_url = self._resolve_connection_url(connection_url)
        self.backend = "sqlite"
        self.is_cloud_persistent = False
        self.is_operational = False
        self.is_fallback = False
        self.init_error = None

        if self.raw_url and (self.raw_url.startswith("postgres://") or self.raw_url.startswith("postgresql://")):
            # Standardize postgres:// to postgresql:// for psycopg2
            self.pg_url = self.raw_url.replace("postgres://", "postgresql://", 1)
            if not PSYCOPG2_AVAILABLE:
                self.init_error = "psycopg2 library not available in environment."
                self.is_fallback = True
                self._init_sqlite()
            else:
                try:
                    self._test_pg_connection()
                    self.backend = "postgresql"
                    self.is_cloud_persistent = True
                    self._init_postgresql()
                    self.is_operational = True
                except Exception as e:
                    logger.warning(f"PostgreSQL connection to {self.get_sanitized_remote_target()} failed: {e}. Falling back to SQLite.")
                    self.init_error = str(e)
                    self.is_fallback = True
                    self._init_sqlite()
        else:
            self._init_sqlite()

    @staticmethod
    def _resolve_connection_url(custom_url: Optional[str] = None) -> Optional[str]:
        if custom_url and custom_url.strip():
            return custom_url.strip()
        # Check environment variables
        for key in ["PERSISTENT_DATABASE_URL", "DATABASE_URL", "SUPABASE_DB_URL", "NEON_DB_URL"]:
            val = os.getenv(key)
            if val and val.strip():
                return val.strip()
        # Check Streamlit secrets if running inside Streamlit
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for key in ["PERSISTENT_DATABASE_URL", "DATABASE_URL", "SUPABASE_DB_URL", "NEON_DB_URL"]:
                    if key in st.secrets:
                        s_val = str(st.secrets[key]).strip()
                        if s_val:
                            return s_val
        except Exception:
            pass
        return None

    def get_sanitized_remote_target(self) -> Optional[str]:
        """Returns the masked version of any configured remote database URL."""
        target_url = getattr(self, "pg_url", None) or self.raw_url
        if target_url and (target_url.startswith("postgres://") or target_url.startswith("postgresql://")):
            try:
                parsed = urlparse(target_url)
                netloc = parsed.hostname or "unknown-host"
                if parsed.port:
                    netloc += f":{parsed.port}"
                user = parsed.username or "postgres"
                return f"{parsed.scheme}://{user}:****@{netloc}{parsed.path}"
            except Exception:
                return "postgresql://[redacted]"
        return None

    def get_sanitized_target(self) -> str:
        """Returns active connection target with passwords masked for safe display in logs and UI."""
        if self.backend == "postgresql":
            return self.get_sanitized_remote_target() or "postgresql://[redacted]"
        return f"sqlite:///{self.db_path.as_posix()}"

    def get_storage_status(self) -> Dict[str, Any]:
        """Provides diagnostic metadata regarding active storage configuration."""
        return {
            'backend': self.backend,
            'is_cloud_persistent': self.is_cloud_persistent,
            'is_operational': self.is_operational,
            'connection_target': self.get_sanitized_target(),
            'configured_remote_target': self.get_sanitized_remote_target(),
            'init_error': self.init_error,
            'is_fallback': self.is_fallback,
            'has_configured_url': bool(self.raw_url)
        }

    def _test_pg_connection(self) -> None:
        conn = psycopg2.connect(self.pg_url, connect_timeout=8)
        conn.close()

    @contextmanager
    def _managed_connection(self):
        """Context manager yielding an active database connection and ensuring explicit closure."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_postgresql(self) -> None:
        with self._managed_connection() as conn:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_logs (
                        run_id VARCHAR(100) PRIMARY KEY,
                        source_name VARCHAR(100) NOT NULL,
                        start_time VARCHAR(50) NOT NULL,
                        end_time VARCHAR(50) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        records_retrieved INTEGER DEFAULT 0,
                        records_stored INTEGER DEFAULT 0,
                        records_duplicate INTEGER DEFAULT 0,
                        records_rejected INTEGER DEFAULT 0,
                        error_message TEXT
                    );
                    """)
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS external_observations (
                        id SERIAL PRIMARY KEY,
                        source_name VARCHAR(100) NOT NULL,
                        source_record_id VARCHAR(255) NOT NULL,
                        observation_timestamp VARCHAR(50) NOT NULL,
                        retrieval_timestamp VARCHAR(50) NOT NULL,
                        observation_type VARCHAR(50) NOT NULL,
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        raw_payload_json TEXT,
                        ch4_value DOUBLE PRECISION,
                        ch4_unit VARCHAR(50),
                        co2_value DOUBLE PRECISION,
                        co2_unit VARCHAR(50),
                        o2_value DOUBLE PRECISION,
                        o2_unit VARCHAR(50),
                        auxiliary_metrics_json TEXT,
                        is_classifier_compatible INTEGER NOT NULL DEFAULT 0,
                        incompatibility_reason TEXT,
                        data_quality_status VARCHAR(50) NOT NULL DEFAULT 'VERIFIED_PUBLIC',
                        is_synthetic INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT uq_source_record UNIQUE(source_name, source_record_id)
                    );
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_source ON external_observations(source_name);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON external_observations(observation_timestamp);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_compatible ON external_observations(is_classifier_compatible);")

    def _init_sqlite(self) -> None:
        self.backend = "sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._managed_connection() as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_logs (
                        run_id TEXT PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        status TEXT NOT NULL,
                        records_retrieved INTEGER DEFAULT 0,
                        records_stored INTEGER DEFAULT 0,
                        records_duplicate INTEGER DEFAULT 0,
                        records_rejected INTEGER DEFAULT 0,
                        error_message TEXT
                    );
                    """)
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS external_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT NOT NULL,
                        source_record_id TEXT NOT NULL,
                        observation_timestamp TEXT NOT NULL,
                        retrieval_timestamp TEXT NOT NULL,
                        observation_type TEXT NOT NULL,
                        latitude REAL,
                        longitude REAL,
                        raw_payload_json TEXT,
                        ch4_value REAL,
                        ch4_unit TEXT,
                        co2_value REAL,
                        co2_unit TEXT,
                        o2_value REAL,
                        o2_unit TEXT,
                        auxiliary_metrics_json TEXT,
                        is_classifier_compatible INTEGER NOT NULL DEFAULT 0,
                        incompatibility_reason TEXT,
                        data_quality_status TEXT NOT NULL DEFAULT 'VERIFIED_PUBLIC',
                        is_synthetic INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now')),
                        UNIQUE(source_name, source_record_id)
                    );
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_source ON external_observations(source_name);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON external_observations(observation_timestamp);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_compatible ON external_observations(is_classifier_compatible);")
            self.is_operational = True
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            self.init_error = str(e)
            self.is_operational = False

    def get_connection(self):
        """Returns active database connection."""
        if self.backend == "postgresql":
            return psycopg2.connect(self.pg_url, connect_timeout=10)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def record_ingestion_run(self, summary: IngestionSummary) -> None:
        """Records the results of an ingestion run into the audit log."""
        if self.backend == "postgresql":
            with self._managed_connection() as conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                        INSERT INTO ingestion_logs (
                            run_id, source_name, start_time, end_time, status,
                            records_retrieved, records_stored, records_duplicate, records_rejected, error_message
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id) DO UPDATE SET
                            start_time = EXCLUDED.start_time,
                            end_time = EXCLUDED.end_time,
                            status = EXCLUDED.status,
                            records_retrieved = EXCLUDED.records_retrieved,
                            records_stored = EXCLUDED.records_stored,
                            records_duplicate = EXCLUDED.records_duplicate,
                            records_rejected = EXCLUDED.records_rejected,
                            error_message = EXCLUDED.error_message;
                        """, (
                            summary.run_id, summary.source_name, summary.start_time, summary.end_time,
                            summary.status, summary.records_retrieved, summary.records_stored,
                            summary.records_duplicate, summary.records_rejected, summary.error_message
                        ))
            return

        with self._managed_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO ingestion_logs (
                    run_id, source_name, start_time, end_time, status,
                    records_retrieved, records_stored, records_duplicate, records_rejected, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    summary.run_id, summary.source_name, summary.start_time, summary.end_time,
                    summary.status, summary.records_retrieved, summary.records_stored,
                    summary.records_duplicate, summary.records_rejected, summary.error_message
                ))

    def store_observations(self, observations: List[NormalizedObservation]) -> Tuple[int, int, int]:
        """
        Idempotently stores normalized observations.
        Returns: (stored_count, duplicate_count, rejected_count)
        """
        stored = 0
        duplicates = 0
        rejected = 0

        if self.backend == "postgresql":
            with self._managed_connection() as conn:
                with conn:
                    with conn.cursor() as cursor:
                        for obs in observations:
                            try:
                                cursor.execute("SAVEPOINT obs_savepoint;")
                                cursor.execute("""
                                INSERT INTO external_observations (
                                    source_name, source_record_id, observation_timestamp, retrieval_timestamp,
                                    observation_type, latitude, longitude, raw_payload_json,
                                    ch4_value, ch4_unit, co2_value, co2_unit, o2_value, o2_unit,
                                    auxiliary_metrics_json, is_classifier_compatible, incompatibility_reason,
                                    data_quality_status, is_synthetic
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (source_name, source_record_id) DO NOTHING
                                RETURNING id;
                                """, (
                                    obs.source_name, obs.source_record_id, obs.observation_timestamp,
                                    obs.retrieval_timestamp, obs.observation_type, obs.latitude,
                                    obs.longitude, obs.raw_payload_json, obs.ch4_value, obs.ch4_unit,
                                    obs.co2_value, obs.co2_unit, obs.o2_value, obs.o2_unit,
                                    obs.auxiliary_metrics_json, 1 if obs.is_classifier_compatible else 0,
                                    obs.incompatibility_reason, obs.data_quality_status,
                                    1 if obs.is_synthetic else 0
                                ))
                                row = cursor.fetchone()
                                if row is not None:
                                    stored += 1
                                else:
                                    duplicates += 1
                                cursor.execute("RELEASE SAVEPOINT obs_savepoint;")
                            except Exception as e:
                                cursor.execute("ROLLBACK TO SAVEPOINT obs_savepoint;")
                                logger.warning(f"Error storing record {obs.source_record_id} in PostgreSQL: {e}")
                                rejected += 1
            return stored, duplicates, rejected

        # SQLite branch
        with self._managed_connection() as conn:
            with conn:
                cursor = conn.cursor()
                for obs in observations:
                    try:
                        cursor.execute("""
                        INSERT INTO external_observations (
                            source_name, source_record_id, observation_timestamp, retrieval_timestamp,
                            observation_type, latitude, longitude, raw_payload_json,
                            ch4_value, ch4_unit, co2_value, co2_unit, o2_value, o2_unit,
                            auxiliary_metrics_json, is_classifier_compatible, incompatibility_reason,
                            data_quality_status, is_synthetic
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            obs.source_name, obs.source_record_id, obs.observation_timestamp,
                            obs.retrieval_timestamp, obs.observation_type, obs.latitude,
                            obs.longitude, obs.raw_payload_json, obs.ch4_value, obs.ch4_unit,
                            obs.co2_value, obs.co2_unit, obs.o2_value, obs.o2_unit,
                            obs.auxiliary_metrics_json, 1 if obs.is_classifier_compatible else 0,
                            obs.incompatibility_reason, obs.data_quality_status,
                            1 if obs.is_synthetic else 0
                        ))
                        stored += 1
                    except sqlite3.IntegrityError as e:
                        if "UNIQUE constraint failed" in str(e):
                            duplicates += 1
                        else:
                            logger.warning(f"Rejected record {obs.source_record_id}: {e}")
                            rejected += 1
                    except Exception as e:
                        logger.warning(f"Error storing record {obs.source_record_id}: {e}")
                        rejected += 1

        return stored, duplicates, rejected

    def get_latest_ingestion_summary(self) -> Optional[Dict[str, Any]]:
        """Returns metadata regarding the most recent ingestion run."""
        if self.backend == "postgresql":
            try:
                with self._managed_connection() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                        cursor.execute("SELECT * FROM ingestion_logs ORDER BY start_time DESC LIMIT 1;")
                        row = cursor.fetchone()
                        if row:
                            return dict(row)
            except Exception as e:
                logger.warning(f"Error fetching latest summary from PostgreSQL: {e}")
            return None

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM ingestion_logs
            ORDER BY start_time DESC
            LIMIT 1;
            """)
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_latest_successful_sync_time(self) -> Optional[str]:
        """Returns the UTC timestamp of the latest successful ingestion run."""
        if self.backend == "postgresql":
            try:
                with self._managed_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT end_time FROM ingestion_logs WHERE status = 'SUCCESS' ORDER BY end_time DESC LIMIT 1;")
                        row = cursor.fetchone()
                        if row and row[0]:
                            return str(row[0])
            except Exception as e:
                logger.warning(f"Error fetching latest sync time from PostgreSQL: {e}")
            return None

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT end_time FROM ingestion_logs
            WHERE status = 'SUCCESS'
            ORDER BY end_time DESC
            LIMIT 1;
            """)
            row = cursor.fetchone()
            if row and row['end_time']:
                return str(row['end_time'])
        return None

    def get_latest_observation_timestamp(self) -> Optional[str]:
        """Returns the UTC timestamp of the newest observation stored in the database."""
        if self.backend == "postgresql":
            try:
                with self._managed_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT MAX(observation_timestamp) FROM external_observations;")
                        row = cursor.fetchone()
                        if row and row[0]:
                            return str(row[0])
            except Exception as e:
                logger.warning(f"Error fetching latest observation timestamp from PostgreSQL: {e}")
            return None

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(observation_timestamp) FROM external_observations;")
            row = cursor.fetchone()
            if row and row[0]:
                return str(row[0])
        return None

    def get_observations_dataframe(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves stored external observations as a pandas DataFrame."""
        if self.backend == "postgresql":
            try:
                query = """
                SELECT
                    id, source_name, source_record_id, observation_timestamp,
                    retrieval_timestamp, observation_type, latitude, longitude,
                    ch4_value, ch4_unit, co2_value, co2_unit, o2_value, o2_unit,
                    is_classifier_compatible, incompatibility_reason, data_quality_status, is_synthetic
                FROM external_observations
                ORDER BY observation_timestamp DESC
                LIMIT %s;
                """
                with self._managed_connection() as conn:
                    df = pd.read_sql_query(query, conn, params=(limit,))
                return df
            except Exception as e:
                logger.warning(f"Error querying PostgreSQL observations: {e}")
                return pd.DataFrame()

        with self._managed_connection() as conn:
            query = """
            SELECT
                id, source_name, source_record_id, observation_timestamp,
                retrieval_timestamp, observation_type, latitude, longitude,
                ch4_value, ch4_unit, co2_value, co2_unit, o2_value, o2_unit,
                is_classifier_compatible, incompatibility_reason, data_quality_status, is_synthetic
            FROM external_observations
            ORDER BY observation_timestamp DESC
            LIMIT ?;
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df

    def get_observation_counts(self) -> Dict[str, int]:
        """Returns total, compatible, incompatible, and synthetic observation counts."""
        if self.backend == "postgresql":
            try:
                with self._managed_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM external_observations;")
                        total = cursor.fetchone()[0] or 0
                        cursor.execute("SELECT COUNT(*) FROM external_observations WHERE is_classifier_compatible = 1;")
                        compatible = cursor.fetchone()[0] or 0
                        cursor.execute("SELECT COUNT(*) FROM external_observations WHERE is_synthetic = 1;")
                        synthetic = cursor.fetchone()[0] or 0
                        cursor.execute("SELECT COUNT(*) FROM external_observations WHERE data_quality_status = 'VERIFIED_PUBLIC';")
                        verified_public = cursor.fetchone()[0] or 0
                return {
                    'total': total,
                    'compatible': compatible,
                    'incompatible': total - compatible,
                    'synthetic': synthetic,
                    'verified_public': verified_public
                }
            except Exception as e:
                logger.warning(f"Error fetching counts from PostgreSQL: {e}")
                return {'total': 0, 'compatible': 0, 'incompatible': 0, 'synthetic': 0, 'verified_public': 0}

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM external_observations;")
            total = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM external_observations WHERE is_classifier_compatible = 1;")
            compatible = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM external_observations WHERE is_synthetic = 1;")
            synthetic = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM external_observations WHERE data_quality_status = 'VERIFIED_PUBLIC';")
            verified_public = cursor.fetchone()[0] or 0

        return {
            'total': total,
            'compatible': compatible,
            'incompatible': total - compatible,
            'synthetic': synthetic,
            'verified_public': verified_public
        }

    def get_ingestion_logs_dataframe(self, limit: int = 20) -> pd.DataFrame:
        """Retrieves recent ingestion audit logs as a pandas DataFrame."""
        if self.backend == "postgresql":
            try:
                query = """
                SELECT run_id, source_name, start_time, end_time, status,
                       records_retrieved, records_stored, records_duplicate, records_rejected, error_message
                FROM ingestion_logs
                ORDER BY start_time DESC
                LIMIT %s;
                """
                with self._managed_connection() as conn:
                    df = pd.read_sql_query(query, conn, params=(limit,))
                return df
            except Exception as e:
                logger.warning(f"Error querying PostgreSQL logs: {e}")
                return pd.DataFrame()

        with self._managed_connection() as conn:
            query = """
            SELECT run_id, source_name, start_time, end_time, status,
                   records_retrieved, records_stored, records_duplicate, records_rejected, error_message
            FROM ingestion_logs
            ORDER BY start_time DESC
            LIMIT ?;
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df


# ==============================================================================
# SCIENTIFIC VALIDATION & UNIT CHECKING
# ==============================================================================

def validate_classifier_compatibility(
    ch4_val: Optional[float],
    ch4_unit: Optional[str],
    co2_val: Optional[float],
    co2_unit: Optional[str],
    o2_val: Optional[float],
    o2_unit: Optional[str],
    obs_type: str = "unknown"
) -> Tuple[bool, Optional[str]]:
    """
    Rigorously validates whether incoming measurements are scientifically compatible
    with the 3-gas downhole thermal risk classifier (CH4 %, CO2 %, O2 %).

    SCIENTIFIC INTEGRITY RULES:
    1. The model strictly requires all 3 gases: CH4, CO2, and O2.
    2. Units must be volumetric percentage ('Vol %' or '%').
    3. Plume emission flux (kg/h), column concentrations (ppm/ppb), or weather
       variables (temperature, humidity, pressure) are physically incompatible
       and must be rejected with an explicit explanation.
    4. Missing gases must NEVER be synthesized or imputed with dummy constants.
    """
    reasons = []

    if obs_type in ['satellite_plume', 'ambient_meteorology', 'facility_regulatory']:
        return False, (
            f"Observation type '{obs_type}' does not represent downhole wellhead gas concentrations. "
            "Downhole thermal risk classifiers cannot process ambient weather or plume emission rates."
        )

    # Check for missing components
    if ch4_val is None:
        reasons.append("Missing CH4 measurement")
    if co2_val is None:
        reasons.append("Missing CO2 measurement")
    if o2_val is None:
        reasons.append("Missing O2 measurement")

    if reasons:
        return False, f"Incomplete gas composition: {', '.join(reasons)}. All 3 gases (CH4, CO2, O2) required."

    # Check unit compatibility
    valid_vol_units = ['vol %', '%', 'percentage', 'vol_pct']
    if not ch4_unit or ch4_unit.strip().lower() not in valid_vol_units:
        reasons.append(f"Incompatible CH4 unit '{ch4_unit}' (requires Vol %)")
    if not co2_unit or co2_unit.strip().lower() not in valid_vol_units:
        reasons.append(f"Incompatible CO2 unit '{co2_unit}' (requires Vol %)")
    if not o2_unit or o2_unit.strip().lower() not in valid_vol_units:
        reasons.append(f"Incompatible O2 unit '{o2_unit}' (requires Vol %)")

    if reasons:
        return False, f"Unit incompatibility: {'; '.join(reasons)}. Satellite flux (kg/h) or ppm cannot be converted to Vol %."

    # Check physical measurement ranges (0-100% Vol %)
    if not (0.0 <= ch4_val <= 100.0):
        reasons.append(f"CH4 {ch4_val} out of physical range [0, 100]")
    if not (0.0 <= co2_val <= 100.0):
        reasons.append(f"CO2 {co2_val} out of physical range [0, 100]")
    if not (0.0 <= o2_val <= 25.0):
        reasons.append(f"O2 {o2_val} out of physical range [0, 25]")

    if reasons:
        return False, f"Range violation: {'; '.join(reasons)}."

    return True, None


# ==============================================================================
# SOURCE ADAPTERS
# ==============================================================================

class BaseSourceAdapter(ABC):
    """Abstract base class for all external environmental source connectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier name for this data source."""
        pass

    @property
    @abstractmethod
    def source_citation(self) -> str:
        """Formal citation, licensing, or documentation URL for this source."""
        pass

    @abstractmethod
    def fetch_records(self, max_records: int = 50) -> List[Dict[str, Any]]:
        """Fetches raw observations from the source with network timeouts and retries."""
        pass

    @abstractmethod
    def normalize_record(self, raw_record: Dict[str, Any]) -> NormalizedObservation:
        """Converts raw source payload into a standardized NormalizedObservation."""
        pass


class EPAEnvirofactsAdapter(BaseSourceAdapter):
    """
    Verified public connector for the US EPA Envirofacts Greenhouse Gas Reporting Program (GHGRP) API.
    Retrieves annual regulatory reporting records for Bridgeton Landfill (EPA Facility ID: 1007026).
    Completely open public API; requires no API key.
    """

    API_URL = "https://data.epa.gov/efservice/PUB_DIM_FACILITY/facility_id/1007026/JSON"
    BRIDGETON_FACILITY_ID = 1007026

    @property
    def source_name(self) -> str:
        return "US_EPA_Envirofacts_GHGRP"

    @property
    def source_citation(self) -> str:
        return "US Environmental Protection Agency (EPA) Envirofacts Data Service. Public Domain. https://data.epa.gov/efservice/"

    def fetch_records(self, max_records: int = 50) -> List[Dict[str, Any]]:
        """Queries EPA Envirofacts with exponential backoff and timeout."""
        retries = 3
        backoff = 1.5
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Connecting to EPA Envirofacts API (attempt {attempt}/{retries})...")
                response = requests.get(self.API_URL, timeout=12.0)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        return data[:max_records]
                    return []
                elif response.status_code == 429:
                    logger.warning("EPA Envirofacts returned rate limit HTTP 429. Backing off...")
                    time.sleep(backoff * attempt)
                else:
                    logger.warning(f"EPA Envirofacts returned HTTP {response.status_code}: {response.text[:200]}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"EPA Envirofacts network error on attempt {attempt}: {e}")
                if attempt == retries:
                    raise ConnectionError(f"EPA Envirofacts unavailable after {retries} attempts: {e}")
                time.sleep(backoff * attempt)
        return []

    def normalize_record(self, raw_record: Dict[str, Any]) -> NormalizedObservation:
        retrieval_now = datetime.now(timezone.utc).isoformat()
        year = raw_record.get('year', 2023)
        obs_timestamp = f"{year}-12-31T23:59:59Z"
        record_id = f"EPA_GHGRP_{raw_record.get('facility_id', self.BRIDGETON_FACILITY_ID)}_{year}_{raw_record.get('submission_id', '0')}"

        is_compatible, reason = validate_classifier_compatibility(
            ch4_val=None, ch4_unit=None,
            co2_val=None, co2_unit=None,
            o2_val=None, o2_unit=None,
            obs_type='facility_regulatory'
        )

        aux = {
            'facility_name': raw_record.get('facility_name'),
            'reported_subparts': raw_record.get('reported_subparts'),
            'parent_company': raw_record.get('parent_company'),
            'reporting_year': year,
            'naics_code': raw_record.get('naics_code')
        }

        return NormalizedObservation(
            source_name=self.source_name,
            source_record_id=record_id,
            observation_timestamp=obs_timestamp,
            retrieval_timestamp=retrieval_now,
            observation_type='facility_regulatory',
            latitude=float(raw_record.get('latitude', 38.77)),
            longitude=float(raw_record.get('longitude', -90.445269)),
            raw_payload_json=json.dumps(raw_record),
            ch4_value=None,
            ch4_unit=None,
            co2_value=None,
            co2_unit=None,
            o2_value=None,
            o2_unit=None,
            auxiliary_metrics_json=json.dumps(aux),
            is_classifier_compatible=is_compatible,
            incompatibility_reason=reason,
            data_quality_status='VERIFIED_PUBLIC',
            is_synthetic=False
        )


class OpenMeteoEnvironmentalAdapter(BaseSourceAdapter):
    """
    Verified public connector for live ambient meteorological observations at Bridgeton Landfill.
    Coordinates: 38.77° N, -90.44° W (Bridgeton quarry vicinity).
    Provides real-time ambient temperature, barometric pressure, and relative humidity.
    Completely open public API; requires no API key.
    """

    API_URL = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=38.77&longitude=-90.44"
        "&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    )

    @property
    def source_name(self) -> str:
        return "Open_Meteo_Ambient_Weather"

    @property
    def source_citation(self) -> str:
        return "Open-Meteo Weather API. Non-commercial open access. https://open-meteo.com/"

    def fetch_records(self, max_records: int = 50) -> List[Dict[str, Any]]:
        """Queries Open-Meteo API for current ambient surface conditions."""
        retries = 3
        backoff = 1.0
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Fetching ambient weather from Open-Meteo (attempt {attempt}/{retries})...")
                response = requests.get(self.API_URL, timeout=10.0)
                if response.status_code == 200:
                    payload = response.json()
                    current = payload.get("current", {})
                    current["latitude"] = payload.get("latitude", 38.77)
                    current["longitude"] = payload.get("longitude", -90.44)
                    return [current]
                time.sleep(backoff * attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Open-Meteo network error on attempt {attempt}: {e}")
                if attempt == retries:
                    raise ConnectionError(f"Open-Meteo unavailable after {retries} attempts: {e}")
                time.sleep(backoff * attempt)
        return []

    def normalize_record(self, raw_record: Dict[str, Any]) -> NormalizedObservation:
        retrieval_now = datetime.now(timezone.utc).isoformat()
        obs_time = raw_record.get("time", retrieval_now)
        if not obs_time.endswith("Z") and "+" not in obs_time:
            obs_time = f"{obs_time}Z"

        record_id = f"METEO_BRIDGETON_{obs_time.replace(':', '').replace('-', '')}"

        is_compatible, reason = validate_classifier_compatibility(
            ch4_val=None, ch4_unit=None,
            co2_val=None, co2_unit=None,
            o2_val=None, o2_unit=None,
            obs_type='ambient_meteorology'
        )

        aux = {
            'ambient_temperature_c': raw_record.get('temperature_2m'),
            'relative_humidity_percent': raw_record.get('relative_humidity_2m'),
            'surface_pressure_hpa': raw_record.get('surface_pressure'),
            'wind_speed_kmh': raw_record.get('wind_speed_10m')
        }

        return NormalizedObservation(
            source_name=self.source_name,
            source_record_id=record_id,
            observation_timestamp=obs_time,
            retrieval_timestamp=retrieval_now,
            observation_type='ambient_meteorology',
            latitude=float(raw_record.get('latitude', 38.77)),
            longitude=float(raw_record.get('longitude', -90.44)),
            raw_payload_json=json.dumps(raw_record),
            ch4_value=None,
            ch4_unit=None,
            co2_value=None,
            co2_unit=None,
            o2_value=None,
            o2_unit=None,
            auxiliary_metrics_json=json.dumps(aux),
            is_classifier_compatible=is_compatible,
            incompatibility_reason=reason,
            data_quality_status='VERIFIED_PUBLIC',
            is_synthetic=False
        )


class CarbonMapperPlumeAdapter(BaseSourceAdapter):
    """
    Connector for satellite/airborne methane plume detections (Carbon Mapper / GHGSat catalog).
    Supports live Carbon Mapper API if CARBON_MAPPER_API_KEY environment variable is configured;
    falls back to confirmed local catalog 'Datasets/Landfill_plume_detections.csv'.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CARBON_MAPPER_API_KEY")

    @property
    def source_name(self) -> str:
        return "Carbon_Mapper_Plume_Catalog"

    @property
    def source_citation(self) -> str:
        return "Carbon Mapper / NASA JPL Airborne Methane Plume Survey. CC-BY 4.0. https://data.carbonmapper.org/"

    def fetch_records(self, max_records: int = 50) -> List[Dict[str, Any]]:
        # If API key configured, attempt live REST endpoint
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            try:
                url = "https://api.carbonmapper.org/api/v1/catalog/plumes?limit=" + str(max_records)
                resp = requests.get(url, headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    return resp.json().get("items", [])[:max_records]
            except Exception as e:
                logger.warning(f"Live Carbon Mapper API failed: {e}. Falling back to confirmed catalog.")

        # Read confirmed local plume catalog
        catalog_path = PROJECT_ROOT / "Datasets" / "Landfill_plume_detections.csv"
        if catalog_path.exists():
            df = pd.read_csv(catalog_path, nrows=max_records)
            return df.to_dict(orient="records")

        return []

    def normalize_record(self, raw_record: Dict[str, Any]) -> NormalizedObservation:
        retrieval_now = datetime.now(timezone.utc).isoformat()

        # Extract plume candidate ID and detection timestamp if encoded in ID
        raw_id = str(raw_record.get('Plume Candidate ID (UTC time of detection)', 'PLUME_UNKNOWN'))
        record_id = f"PLUME_{raw_id.replace(' ', '_').replace(':', '')}"

        obs_time = retrieval_now
        m = re.search(r'(\d{4})(\d{2})(\d{2})t(\d{2})(\d{2})(\d{2})', raw_id, re.IGNORECASE)
        if m:
            obs_time = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}Z"

        # Plume detection emission rate in kg/h
        flux_kg_h = raw_record.get('Instantaneous Emission Rate (kg/h)')
        uncertainty = raw_record.get('Instantaneous Emission Rate Uncertainty (kg/h)')

        is_compatible, reason = validate_classifier_compatibility(
            ch4_val=float(flux_kg_h) if pd.notnull(flux_kg_h) else None,
            ch4_unit="kg/h",
            co2_val=None, co2_unit=None,
            o2_val=None, o2_unit=None,
            obs_type='satellite_plume'
        )

        aux = {
            'ghgrp_facility_id': raw_record.get('GHGRP ID'),
            'instantaneous_emission_rate_kg_h': flux_kg_h if pd.notnull(flux_kg_h) else None,
            'emission_uncertainty_kg_h': uncertainty if pd.notnull(uncertainty) else None,
            'remote_sensing_platform': 'Airborne AVIRIS-NG / Satellite Imaging Spectrometer'
        }

        return NormalizedObservation(
            source_name=self.source_name,
            source_record_id=record_id,
            observation_timestamp=obs_time,
            retrieval_timestamp=retrieval_now,
            observation_type='satellite_plume',
            latitude=float(raw_record.get('Latitude of Plume Origin', 0.0)),
            longitude=float(raw_record.get('Longitude of Plume Origin', 0.0)),
            raw_payload_json=json.dumps({k: v for k, v in raw_record.items() if pd.notnull(v)}),
            ch4_value=float(flux_kg_h) if pd.notnull(flux_kg_h) else None,
            ch4_unit="kg/h",
            co2_value=None,
            co2_unit=None,
            o2_value=None,
            o2_unit=None,
            auxiliary_metrics_json=json.dumps(aux),
            is_classifier_compatible=is_compatible,
            incompatibility_reason=reason,
            data_quality_status='VERIFIED_PUBLIC',
            is_synthetic=False
        )


class MockMonitoringAdapter(BaseSourceAdapter):
    """
    Synthetic fixture adapter strictly designed for offline automated testing.
    Records generated by this adapter are explicitly flagged with is_synthetic=1 and
    data_quality_status='TEST_FIXTURE_DO_NOT_USE_IN_PRODUCTION'.
    """

    @property
    def source_name(self) -> str:
        return "Synthetic_Test_Fixture"

    @property
    def source_citation(self) -> str:
        return "Synthetic unit test fixture for offline pipeline validation. NOT real operational data."

    def fetch_records(self, max_records: int = 5) -> List[Dict[str, Any]]:
        # Generates deterministic mock records for testing
        samples = [
            {
                'id': 'MOCK-EXT-001',
                'timestamp': '2026-10-25T12:00:00Z',
                'type': 'external_wellhead',
                'ch4': 45.2, 'ch4_unit': 'Vol %',
                'co2': 38.1, 'co2_unit': 'Vol %',
                'o2': 0.8, 'o2_unit': 'Vol %',
                'lat': 38.771, 'lon': -90.442
            },
            {
                'id': 'MOCK-EXT-002',
                'timestamp': '2026-10-25T12:15:00Z',
                'type': 'satellite_plume',
                'ch4': 1420.0, 'ch4_unit': 'kg/h',
                'co2': None, 'co2_unit': None,
                'o2': None, 'o2_unit': None,
                'lat': 38.775, 'lon': -90.448
            },
            {
                'id': 'MOCK-EXT-003',
                'timestamp': '2026-10-25T12:30:00Z',
                'type': 'external_wellhead',
                'ch4': 28.5, 'ch4_unit': 'Vol %',
                'co2': 55.2, 'co2_unit': 'Vol %',
                'o2': 1.2, 'o2_unit': 'Vol %',
                'lat': 38.772, 'lon': -90.441
            }
        ]
        return samples[:max_records]

    def normalize_record(self, raw_record: Dict[str, Any]) -> NormalizedObservation:
        retrieval_now = datetime.now(timezone.utc).isoformat()
        obs_type = raw_record.get('type', 'external_wellhead')

        is_compatible, reason = validate_classifier_compatibility(
            ch4_val=raw_record.get('ch4'),
            ch4_unit=raw_record.get('ch4_unit'),
            co2_val=raw_record.get('co2'),
            co2_unit=raw_record.get('co2_unit'),
            o2_val=raw_record.get('o2'),
            o2_unit=raw_record.get('o2_unit'),
            obs_type=obs_type
        )

        return NormalizedObservation(
            source_name=self.source_name,
            source_record_id=str(raw_record.get('id')),
            observation_timestamp=raw_record.get('timestamp', retrieval_now),
            retrieval_timestamp=retrieval_now,
            observation_type=obs_type,
            latitude=raw_record.get('lat'),
            longitude=raw_record.get('lon'),
            raw_payload_json=json.dumps(raw_record),
            ch4_value=raw_record.get('ch4'),
            ch4_unit=raw_record.get('ch4_unit'),
            co2_value=raw_record.get('co2'),
            co2_unit=raw_record.get('co2_unit'),
            o2_value=raw_record.get('o2'),
            o2_unit=raw_record.get('o2_unit'),
            auxiliary_metrics_json=json.dumps({'test_fixture_note': 'Generated for unit testing'}),
            is_classifier_compatible=is_compatible,
            incompatibility_reason=reason,
            data_quality_status='TEST_FIXTURE_DO_NOT_USE_IN_PRODUCTION',
            is_synthetic=True
        )


# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================

def get_registered_adapter(source_name: str) -> BaseSourceAdapter:
    """Returns the adapter instance corresponding to the registered source name."""
    adapters = {
        'US_EPA_Envirofacts_GHGRP': EPAEnvirofactsAdapter,
        'Open_Meteo_Ambient_Weather': OpenMeteoEnvironmentalAdapter,
        'Carbon_Mapper_Plume_Catalog': CarbonMapperPlumeAdapter,
        'Synthetic_Test_Fixture': MockMonitoringAdapter
    }
    if source_name not in adapters:
        raise ValueError(f"Unknown source '{source_name}'. Choose from {list(adapters.keys())}.")
    return adapters[source_name]()


def sync_external_data(
    source_name: str = "US_EPA_Envirofacts_GHGRP",
    max_records: int = 50,
    db_manager: Optional[DatabaseManager] = None
) -> IngestionSummary:
    """
    Executes a complete, safe, and idempotent ingestion cycle for the specified source.

    Steps:
    1. Instantiates the appropriate adapter.
    2. Fetches records with timeouts and retries.
    3. Normalizes timestamps, coordinates, and units.
    4. Enforces 3-gas classifier validation rules.
    5. Deduplicates and stores into SQLite.
    6. Logs audit metadata into ingestion_logs.
    """
    if db_manager is None:
        db_manager = DatabaseManager()

    start_time = datetime.now(timezone.utc).isoformat()
    run_id = f"RUN_{int(time.time())}_{source_name[:12]}"

    try:
        adapter = get_registered_adapter(source_name)
        logger.info(f"Starting ingestion run {run_id} for source {source_name}...")
        raw_records = adapter.fetch_records(max_records=max_records)
        records_retrieved = len(raw_records)

        normalized_obs: List[NormalizedObservation] = []
        for r in raw_records:
            try:
                norm = adapter.normalize_record(r)
                normalized_obs.append(norm)
            except Exception as e:
                logger.warning(f"Error normalizing record in {source_name}: {e}")

        stored, duplicates, rejected = db_manager.store_observations(normalized_obs)
        end_time = datetime.now(timezone.utc).isoformat()

        status = "SUCCESS" if records_retrieved > 0 else "PARTIAL"
        summary = IngestionSummary(
            run_id=run_id,
            source_name=source_name,
            start_time=start_time,
            end_time=end_time,
            status=status,
            records_retrieved=records_retrieved,
            records_stored=stored,
            records_duplicate=duplicates,
            records_rejected=rejected,
            error_message=None
        )
        db_manager.record_ingestion_run(summary)
        logger.info(f"Ingestion run {run_id} completed: retrieved={records_retrieved}, stored={stored}, duplicates={duplicates}.")
        return summary

    except Exception as e:
        end_time = datetime.now(timezone.utc).isoformat()
        err_msg = str(e)
        logger.error(f"Ingestion run {run_id} failed: {err_msg}")
        summary = IngestionSummary(
            run_id=run_id,
            source_name=source_name,
            start_time=start_time,
            end_time=end_time,
            status="FAILED",
            records_retrieved=0,
            records_stored=0,
            records_duplicate=0,
            records_rejected=0,
            error_message=err_msg
        )
        db_manager.record_ingestion_run(summary)
        return summary


if __name__ == "__main__":
    # Command line entrypoint for manual and scheduled execution
    import argparse
    parser = argparse.ArgumentParser(description="Landfill Automated Environmental Ingestion Runner")
    parser.add_argument(
        "--source",
        choices=[
            'US_EPA_Envirofacts_GHGRP',
            'Open_Meteo_Ambient_Weather',
            'Carbon_Mapper_Plume_Catalog',
            'Synthetic_Test_Fixture'
        ],
        default='US_EPA_Envirofacts_GHGRP',
        help="Target data source adapter"
    )
    parser.add_argument("--max-records", type=int, default=50, help="Maximum records to ingest")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database path")
    parser.add_argument("--db-url", type=str, default=None, help="Custom PostgreSQL database connection URL")

    args = parser.parse_args()
    db = DatabaseManager(db_path=args.db_path, connection_url=args.db_url)
    status_diag = db.get_storage_status()
    print("==========================================================")
    print("Storage Backend:", status_diag['backend'])
    print("Connection Target:", status_diag['connection_target'])
    print("Cloud Persistence Active:", status_diag['is_cloud_persistent'])
    if status_diag['init_error']:
        print("Persistence Notice:", status_diag['init_error'])
    print("==========================================================")

    res = sync_external_data(source_name=args.source, max_records=args.max_records, db_manager=db)
    print(json.dumps(asdict(res), indent=2))
    if res.status == "FAILED":
        sys.exit(1)
    sys.exit(0)
