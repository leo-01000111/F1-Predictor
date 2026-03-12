from __future__ import annotations

from src.ui import services


def test_preflight_record_actual_requires_prediction(monkeypatch):
    monkeypatch.setattr(services, "has_prediction", lambda y, r: False)
    monkeypatch.setattr(services, "has_actual_result", lambda y, r: False)

    report = services.get_preflight_report("record_actual", 2026, 3)

    assert report["ready"] is False
    assert report["status"] == "error"
    assert any("Prediction file is missing" in e for e in report["errors"])


def test_preflight_run_inference_warns_if_prediction_exists(monkeypatch):
    monkeypatch.setattr(services, "has_prediction", lambda y, r: True)
    monkeypatch.setattr(services, "has_actual_result", lambda y, r: False)

    report = services.get_preflight_report("run_inference", 2026, 3)

    assert report["ready"] is True
    assert report["status"] == "warn"
    assert any("already exists" in w for w in report["warnings"])


def test_live_widgets_fallback_contains_required_cards():
    pred = {
        "year": 2026,
        "round": 1,
        "circuit_key": "melbourne",
        "weather": {"temp_c_mean": 20.0, "precip_mm_total": 0.0},
        "predictions": [{"driver": "VER", "p_win": 0.55}],
    }
    prior = {
        "predictions": [{"driver": "VER", "p_win": 0.40}],
    }

    payload = services._live_widgets_from_prediction(pred, prior)

    assert payload["source"] == "artifact"
    assert payload["available"] is True
    widgets = payload["widgets"]
    assert "session_state" in widgets
    assert "recent_lap_delta" in widgets
    assert "interval_snapshot" in widgets
    assert "track_condition" in widgets


def test_task_status_chip_maps_expected_states():
    assert "QUEUED" in services.task_status_chip("queued")
    assert "FAILED" in services.task_status_chip("failed")
