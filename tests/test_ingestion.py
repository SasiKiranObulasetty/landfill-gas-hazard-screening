"""
test_ingestion.py - Automated test suite for external environmental ingestion pipeline.
AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills.

Validates all 12 Phase 7 test criteria:
1. Source-response parsing.
2. Schema and unit validation.
3. Missing and malformed fields.
4. Duplicate detection and idempotency.
5. Database initialization and persistence.
6. API timeouts and HTTP errors.
7. Rate limits and retry behavior.
8. Empty responses and unavailable sources.
9. Cache invalidation after successful ingestion.
10. Rejection of incompatible satellite measurements for the existing gas classifier.
11. Preservation of the original historical dataset and trained models.
12. Manual ingestion behavior.

All tests utilize mocked HTTP responses and local SQLite fixtures. Zero live external API calls.
"""

import os
import json
import sqlite3
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import pandas as pd
import requests

from src.ingestion import (
    DatabaseManager,
    NormalizedObservation,
    IngestionSummary,
    validate_classifier_compatibility,
    EPAEnvirofactsAdapter,
    OpenMeteoEnvironmentalAdapter,
    CarbonMapperPlumeAdapter,
    MockMonitoringAdapter,
    sync_external_data
)
from src.data_loader import load_bridgeton_raw
from src.predict import get_model_and_metadata, predict_single
from src.preprocessing import FEATURE_COLS_3GAS


# ==============================================================================
# TEST 1: SOURCE RESPONSE PARSING
# ==============================================================================
def test_source_response_parsing():
    """Verify raw source payloads are correctly parsed and normalized."""
    epa_adapter = EPAEnvirofactsAdapter()
    raw_epa = {
        'facility_id': 1007026,
        'facility_name': 'BRIDGETON LANDFILL',
        'year': 2023,
        'reported_subparts': 'HH,C',
        'latitude': 38.77,
        'longitude': -90.445269,
        'parent_company': 'REPUBLIC SERVICES INC (100%)',
        'submission_id': 99999
    }
    norm_epa = epa_adapter.normalize_record(raw_epa)
    assert norm_epa.source_name == "US_EPA_Envirofacts_GHGRP"
    assert "1007026" in norm_epa.source_record_id
    assert norm_epa.observation_type == "facility_regulatory"
    assert norm_epa.is_classifier_compatible is False
    assert norm_epa.is_synthetic is False

    meteo_adapter = OpenMeteoEnvironmentalAdapter()
    raw_meteo = {
        'time': '2026-10-25T14:00',
        'temperature_2m': 18.5,
        'relative_humidity_2m': 62.0,
        'surface_pressure': 1012.3,
        'wind_speed_10m': 12.4,
        'latitude': 38.77,
        'longitude': -90.44
    }
    norm_meteo = meteo_adapter.normalize_record(raw_meteo)
    assert norm_meteo.source_name == "Open_Meteo_Ambient_Weather"
    assert norm_meteo.observation_type == "ambient_meteorology"
    assert norm_meteo.is_classifier_compatible is False


# ==============================================================================
# TEST 2: SCHEMA AND UNIT VALIDATION (COMPATIBLE WELLHEAD FEED)
# ==============================================================================
def test_schema_and_unit_validation_compatible():
    """Verify that genuine 3-gas volumetric wellhead measurements pass validation."""
    is_valid, reason = validate_classifier_compatibility(
        ch4_val=48.5, ch4_unit="Vol %",
        co2_val=39.2, co2_unit="Vol %",
        o2_val=0.4, o2_unit="Vol %",
        obs_type="external_wellhead"
    )
    assert is_valid is True
    assert reason is None


# ==============================================================================
# TEST 3: MISSING AND MALFORMED FIELDS
# ==============================================================================
def test_missing_and_malformed_fields():
    """Verify that records with missing gases or physically impossible values are rejected."""
    # Missing O2
    is_valid, reason = validate_classifier_compatibility(
        ch4_val=50.0, ch4_unit="Vol %",
        co2_val=45.0, co2_unit="Vol %",
        o2_val=None, o2_unit="Vol %",
        obs_type="external_wellhead"
    )
    assert is_valid is False
    assert "Missing O2" in reason

    # Out-of-range CH4 (> 100%)
    is_valid, reason = validate_classifier_compatibility(
        ch4_val=125.0, ch4_unit="Vol %",
        co2_val=40.0, co2_unit="Vol %",
        o2_val=1.0, o2_unit="Vol %",
        obs_type="external_wellhead"
    )
    assert is_valid is False
    assert "out of physical range" in reason


# ==============================================================================
# TEST 4: DUPLICATE DETECTION AND IDEMPOTENCY
# ==============================================================================
def test_duplicate_detection_and_idempotency(tmp_path):
    """Verify that storing the same observations multiple times generates 0 duplicates."""
    db_file = tmp_path / "test_dedup.db"
    db = DatabaseManager(db_path=db_file)

    mock_adapter = MockMonitoringAdapter()
    records = mock_adapter.fetch_records(max_records=3)
    normalized = [mock_adapter.normalize_record(r) for r in records]

    # First insertion: all stored
    stored1, dup1, rej1 = db.store_observations(normalized)
    assert stored1 == 3
    assert dup1 == 0

    # Second insertion of identical records: all identified as duplicates
    stored2, dup2, rej2 = db.store_observations(normalized)
    assert stored2 == 0
    assert dup2 == 3
    assert rej2 == 0


# ==============================================================================
# TEST 5: DATABASE INITIALIZATION AND PERSISTENCE
# ==============================================================================
def test_database_initialization_and_persistence(tmp_path):
    """Verify SQLite database creates required tables and indices."""
    db_file = tmp_path / "test_init.db"
    db = DatabaseManager(db_path=db_file)
    assert db_file.exists()

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "ingestion_logs" in tables
        assert "external_observations" in tables


# ==============================================================================
# TEST 6: API TIMEOUTS AND HTTP ERRORS
# ==============================================================================
def test_api_timeouts_and_http_errors(tmp_path):
    """Verify that network timeouts and HTTP errors log failures cleanly without crashing."""
    db_file = tmp_path / "test_err.db"
    db = DatabaseManager(db_path=db_file)

    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with patch("time.sleep", return_value=None):
            summary = sync_external_data("US_EPA_Envirofacts_GHGRP", db_manager=db)
            assert summary.status == "FAILED"
            assert "timed out" in summary.error_message
            assert summary.records_stored == 0

    # Ensure failure was recorded in audit log
    latest = db.get_latest_ingestion_summary()
    assert latest['status'] == "FAILED"


# ==============================================================================
# TEST 7: RATE LIMITS AND RETRY BEHAVIOR
# ==============================================================================
def test_rate_limits_and_retry_behavior():
    """Verify that HTTP 429 triggers retry logic with backoff."""
    adapter = EPAEnvirofactsAdapter()

    resp_429 = MagicMock()
    resp_429.status_code = 429

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = [{"facility_id": 1007026, "year": 2023}]

    with patch("requests.get", side_effect=[resp_429, resp_200]):
        with patch("time.sleep", return_value=None):
            records = adapter.fetch_records(max_records=5)
            assert len(records) == 1
            assert records[0]['facility_id'] == 1007026


# ==============================================================================
# TEST 8: EMPTY RESPONSES AND UNAVAILABLE SOURCES
# ==============================================================================
def test_empty_responses_and_unavailable_sources(tmp_path):
    """Verify that empty payloads return PARTIAL status without error."""
    db_file = tmp_path / "test_empty.db"
    db = DatabaseManager(db_path=db_file)

    resp_empty = MagicMock()
    resp_empty.status_code = 200
    resp_empty.json.return_value = []

    with patch("requests.get", return_value=resp_empty):
        summary = sync_external_data("US_EPA_Envirofacts_GHGRP", db_manager=db)
        assert summary.status == "PARTIAL"
        assert summary.records_retrieved == 0
        assert summary.records_stored == 0


# ==============================================================================
# TEST 9: CACHE INVALIDATION BEHAVIOR
# ==============================================================================
def test_cache_invalidation_behavior():
    """Verify external data cache invalidation function exists and leaves model cache intact."""
    with patch("src.data_loader.load_bridgeton_raw", return_value=pd.DataFrame({'a': [1]})):
        from app import get_cached_external_observations, get_cached_model

    # Verify get_cached_external_observations is decorated with cache_data
    assert hasattr(get_cached_external_observations, "clear")

    # Clear external data cache
    get_cached_external_observations.clear()

    # Model cache must remain callable and intact
    model, meta = get_cached_model("random_forest")
    assert model is not None
    assert "models_summary" in meta


# ==============================================================================
# TEST 10: REJECTION OF INCOMPATIBLE SATELLITE MEASUREMENTS
# ==============================================================================
def test_rejection_of_incompatible_satellite_measurements():
    """Verify that satellite plume emission rates (kg/h) are rejected from 3-gas classifier."""
    is_valid, reason = validate_classifier_compatibility(
        ch4_val=1550.0,
        ch4_unit="kg/h",
        co2_val=None,
        co2_unit=None,
        o2_val=None,
        o2_unit=None,
        obs_type="satellite_plume"
    )
    assert is_valid is False
    assert "does not represent downhole wellhead gas concentrations" in reason

    # Verify that predict_single rejects invalid input
    res = predict_single(ch4=1550.0, co2=0.0, o2=0.0)
    assert res['success'] is False
    assert "error" in res


# ==============================================================================
# TEST 11: PRESERVATION OF HISTORICAL DATASET AND MODELS
# ==============================================================================
def test_preservation_of_historical_dataset_and_models():
    """Verify that the original Bridgeton Excel dataset (18,469 rows) and models are untouched."""
    df_raw = load_bridgeton_raw()
    assert len(df_raw) == 18469
    assert list(df_raw.columns) == [
        'Well_Name', 'Date_Sampled', 'CH4_percent', 'CO2_percent',
        'O2_percent', 'Residual_N2_percent', 'Temperature_F',
        'Temperature_C', 'Ratio_CH4_CO2', 'Risk_Category'
    ]

    for model_name in ['random_forest', 'decision_tree', 'extra_trees', 'logistic_regression']:
        model, _ = get_model_and_metadata(model_name)
        assert model is not None


# ==============================================================================
# TEST 12: MANUAL INGESTION BEHAVIOR
# ==============================================================================
def test_manual_ingestion_behavior(tmp_path):
    """Verify synchronous manual ingestion execution via sync_external_data."""
    db_file = tmp_path / "test_manual.db"
    db = DatabaseManager(db_path=db_file)

    summary = sync_external_data("Synthetic_Test_Fixture", max_records=2, db_manager=db)
    assert summary.status == "SUCCESS"
    assert summary.records_retrieved == 2
    assert summary.records_stored == 2

    # Verify DataFrame retrieval
    df_obs = db.get_observations_dataframe()
    assert len(df_obs) == 2
    assert df_obs['is_synthetic'].iloc[0] == 1


# ==============================================================================
# TEST 13: DATABASE MISSING CREDENTIALS FALLBACK
# ==============================================================================
def test_database_missing_credentials_fallback(tmp_path, monkeypatch):
    """Verify that absent credentials cleanly initialize local SQLite without crashing."""
    monkeypatch.delenv("PERSISTENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("NEON_DB_URL", raising=False)

    db_file = tmp_path / "sqlite_test.db"
    db = DatabaseManager(db_path=db_file)

    status = db.get_storage_status()
    assert status['backend'] == "sqlite"
    assert status['is_cloud_persistent'] is False
    assert status['is_fallback'] is False
    assert status['has_configured_url'] is False
    assert status['is_operational'] is True
    assert status['init_error'] is None
    assert "sqlite:///" in status['connection_target']


# ==============================================================================
# TEST 14: DATABASE UNREACHABLE POSTGRESQL FALLBACK
# ==============================================================================
def test_database_unreachable_pg_fallback(tmp_path):
    """Verify that an unreachable PostgreSQL URL falls back to local SQLite with diagnostic logging."""
    db_file = tmp_path / "fallback.db"
    bad_url = "postgresql://test_user:SuperSecretPassword@127.0.0.1:59999/test_db"
    db = DatabaseManager(db_path=db_file, connection_url=bad_url)

    status = db.get_storage_status()
    assert status['backend'] == "sqlite"
    assert status['is_cloud_persistent'] is False
    assert status['is_fallback'] is True
    assert status['has_configured_url'] is True
    assert status['is_operational'] is True
    assert status['init_error'] is not None
    # Password must NEVER be exposed in logs or status
    assert "SuperSecretPassword" not in status['connection_target']
    assert "SuperSecretPassword" not in str(status['configured_remote_target'])
    assert "test_user:****@127.0.0.1:59999/test_db" in status['configured_remote_target']

    # The fallback database must remain fully operational
    mock_adapter = MockMonitoringAdapter()
    records = mock_adapter.fetch_records(max_records=1)
    norm = [mock_adapter.normalize_record(r) for r in records]
    stored, dup, rej = db.store_observations(norm)
    assert stored == 1
    assert rej == 0


# ==============================================================================
# TEST 15: PASSWORD MASKING IN CONNECTION TARGET
# ==============================================================================
def test_database_sanitized_target_password_masking(tmp_path):
    """Verify that credentials are fully masked in get_sanitized_remote_target and active target."""
    db_file = tmp_path / "mask_test.db"
    urls = [
        "postgresql://postgres:MyVerySecretPassword123@ep-cool-fog-12345.us-east-2.aws.neon.tech:5432/neondb",
        "postgres://supabase_admin:P@ssw0rd!#%@db.xyz.supabase.co:6543/postgres"
    ]
    for url in urls:
        db = DatabaseManager(db_path=db_file, connection_url=url)
        remote_target = db.get_sanitized_remote_target()
        assert remote_target is not None
        assert "MyVerySecretPassword123" not in remote_target
        assert "P@ssw0rd!#%" not in remote_target
        assert ":****@" in remote_target

    # When connected to PostgreSQL, active get_sanitized_target() also returns the masked remote target
    with patch("psycopg2.connect", return_value=MagicMock()):
        db_pg = DatabaseManager(db_path=db_file, connection_url="postgresql://admin:TopSecretKey@dbhost:5432/db")
        assert "TopSecretKey" not in db_pg.get_sanitized_target()
        assert "admin:****@dbhost:5432/db" in db_pg.get_sanitized_target()


# ==============================================================================
# TEST 16: MOCKED POSTGRESQL ADAPTER EXECUTION
# ==============================================================================
def test_mocked_postgresql_adapter_execution(tmp_path):
    """Verify PostgreSQL SQL queries, conflict handling, and queries using mocked psycopg2."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    pg_url = "postgresql://pguser:pgpass@remotehost:5432/landfill_db"

    with patch("psycopg2.connect", return_value=mock_conn):
        db = DatabaseManager(db_path=tmp_path / "unused.db", connection_url=pg_url)
        assert db.backend == "postgresql"
        assert db.is_cloud_persistent is True
        assert db.is_operational is True

        # Test record_ingestion_run
        summary = IngestionSummary(
            run_id="run_pg_001",
            source_name="US_EPA_Envirofacts_GHGRP",
            start_time="2026-10-29T00:00:00Z",
            end_time="2026-10-29T00:01:00Z",
            status="SUCCESS",
            records_retrieved=10,
            records_stored=8,
            records_duplicate=2,
            records_rejected=0
        )
        db.record_ingestion_run(summary)
        # Verify ON CONFLICT was included in query
        call_args = mock_cursor.execute.call_args_list
        ingest_sql = [c[0][0] for c in call_args if "INSERT INTO ingestion_logs" in str(c[0][0])]
        assert len(ingest_sql) > 0
        assert "ON CONFLICT (run_id) DO UPDATE" in ingest_sql[0]

        # Test store_observations with new record (returns id) and duplicate record (returns None)
        obs_new = NormalizedObservation(
            source_name="US_EPA_Envirofacts_GHGRP",
            source_record_id="EPA_101",
            observation_timestamp="2026-10-29T00:00:00Z",
            retrieval_timestamp="2026-10-29T00:00:05Z",
            observation_type="facility_regulatory"
        )
        obs_dup = NormalizedObservation(
            source_name="US_EPA_Envirofacts_GHGRP",
            source_record_id="EPA_102",
            observation_timestamp="2026-10-29T00:00:00Z",
            retrieval_timestamp="2026-10-29T00:00:05Z",
            observation_type="facility_regulatory"
        )

        mock_cursor.fetchone.side_effect = [(1,), None]
        stored, duplicates, rejected = db.store_observations([obs_new, obs_dup])
        assert stored == 1
        assert duplicates == 1
        assert rejected == 0

        # Test get_latest_observation_timestamp
        mock_cursor.fetchone.side_effect = [("2026-10-28T18:00:00Z",)]
        latest_obs = db.get_latest_observation_timestamp()
        assert latest_obs == "2026-10-28T18:00:00Z"


# ==============================================================================
# TEST 17: INVALID SOURCE NAME AUDITED AS FAILURE
# ==============================================================================
def test_invalid_source_name_audited_as_failure(tmp_path):
    """Verify that an unknown or invalid source name logs an audited FAILED entry."""
    db = DatabaseManager(db_path=tmp_path / "audit_fail.db")
    summary = sync_external_data("Unknown_Source_ABC", db_manager=db)
    assert summary.status == "FAILED"
    assert summary.error_message is not None
    assert "Unknown_Source_ABC" in summary.error_message

    logs = db.get_ingestion_logs_dataframe()
    assert len(logs) >= 1
    assert logs.iloc[0]['status'] == "FAILED"
    assert "Unknown_Source_ABC" in logs.iloc[0]['error_message']


# ==============================================================================
# TEST 18: CARBON MAPPER DETECTION TIMESTAMP EXTRACTION
# ==============================================================================
def test_carbon_mapper_candidate_timestamp_extraction():
    """Verify that candidate ID timestamps (tYYYYMMDDtHHMMSS) are parsed into observation_timestamp."""
    adapter = CarbonMapperPlumeAdapter()
    raw = {
        'GHGRP ID': 1007026,
        'Latitude of Plume Origin': 38.77,
        'Longitude of Plume Origin': -90.44,
        'Plume Candidate ID (UTC time of detection)': 'GAO20211026t174729p0000-B',
        'Instantaneous Emission Rate (kg/h)': 4500.0,
        'Instantaneous Emission Rate Uncertainty (kg/h)': 250.0
    }
    norm = adapter.normalize_record(raw)
    assert norm.observation_timestamp == "2021-10-26T17:47:29Z"
    assert norm.retrieval_timestamp is not None
    assert norm.observation_type == "satellite_plume"
    assert norm.is_classifier_compatible is False


# ==============================================================================
# TEST 19: POSTGRESQL SAVEPOINT ERROR ISOLATION
# ==============================================================================
def test_postgresql_savepoint_error_isolation(tmp_path):
    """Verify PostgreSQL batch store uses savepoints to prevent InFailedSqlTransaction cascade."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    pg_url = "postgresql://pguser:pgpass@remotehost:5432/landfill_db"

    with patch("psycopg2.connect", return_value=mock_conn):
        db = DatabaseManager(db_path=tmp_path / "unused.db", connection_url=pg_url)

        obs1 = NormalizedObservation(
            source_name="US_EPA_Envirofacts_GHGRP",
            source_record_id="EPA_VALID_1",
            observation_timestamp="2026-10-29T00:00:00Z",
            retrieval_timestamp="2026-10-29T00:00:05Z",
            observation_type="facility_regulatory"
        )
        obs2 = NormalizedObservation(
            source_name="US_EPA_Envirofacts_GHGRP",
            source_record_id="EPA_ERROR_2",
            observation_timestamp="2026-10-29T00:00:00Z",
            retrieval_timestamp="2026-10-29T00:00:05Z",
            observation_type="facility_regulatory"
        )

        # Simulate obs1 success, obs2 query failure
        mock_cursor.execute.side_effect = [
            None, None, None,  # _init_postgresql table & indices
            None,  # SAVEPOINT obs1
            None,  # INSERT obs1
            None,  # RELEASE SAVEPOINT obs1
            None,  # SAVEPOINT obs2
            Exception("Simulated constraint violation on record 2"),  # INSERT obs2 raises
            None   # ROLLBACK TO SAVEPOINT obs2
        ]
        mock_cursor.fetchone.side_effect = [(1,)]

        stored, dup, rej = db.store_observations([obs1, obs2])
        assert stored == 1
        assert dup == 0
        assert rej == 1

        # Check that savepoint and rollback statements were issued
        calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list if c[0]]
        assert any("SAVEPOINT obs_savepoint" in c for c in calls)
        assert any("ROLLBACK TO SAVEPOINT obs_savepoint" in c for c in calls)

