"""
test_app_startup.py - Tests for Streamlit app compilation and navigation consistency.
"""

import ast
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def test_app_syntax_and_compilation():
    """Verify that app.py contains valid, compilable Python AST."""
    assert APP_PATH.exists(), "app.py does not exist at project root!"
    with open(APP_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    
    # Parse AST to guarantee zero syntax errors
    parsed = ast.parse(source, filename="app.py")
    assert isinstance(parsed, ast.Module)


def test_app_contains_all_seven_pages():
    """Verify that all 7 required demonstration pages are registered in app.py."""
    with open(APP_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    required_pages = [
        "Project Overview",
        "Dataset Explorer",
        "Risk-Category Classification",
        "Interactive Gas Screening",
        "Research Visualizations",
        "Seven-Day Forecasting Research",
        "Model Evaluation & Limitations",
        "Data Updates & Monitoring"
    ]

    for page in required_pages:
        assert f'"{page}"' in source or f"'{page}'" in source, f"Required page '{page}' missing from app.py!"


def test_app_import_and_execution_components():
    """Verify that app.py imports cleanly without raising any ImportError."""
    import app
    assert hasattr(app, "create_macro_f1_comparison_figure")
    assert callable(app.create_macro_f1_comparison_figure)
    assert hasattr(app, "create_high_risk_tradeoff_figure")
    assert callable(app.create_high_risk_tradeoff_figure)
    assert hasattr(app, "get_classical_model_comparison_summary")
    assert callable(app.get_classical_model_comparison_summary)
    assert hasattr(app, "get_cached_external_observations")
