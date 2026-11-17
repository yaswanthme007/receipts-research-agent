import json

from fastapi.testclient import TestClient

from receipts import server


def client(monkeypatch, nebius="", tavily=""):
    monkeypatch.setenv("NEBIUS_API_KEY", nebius)
    monkeypatch.setenv("TAVILY_API_KEY", tavily)
    return TestClient(server.app)


def test_health_reports_missing_keys_without_leaking_values(monkeypatch):
    c = client(monkeypatch, nebius="", tavily="tvly-secret-value")
    body = c.get("/api/health").json()
    assert body["ready"] is False
    assert body["missing"] == ["NEBIUS_API_KEY"]
    assert "tvly-secret-value" not in json.dumps(body)


def test_run_without_keys_gives_actionable_error(monkeypatch):
    c = client(monkeypatch)
    r = c.post("/api/run", json={"mode": "research", "input": "What is X?"})
    assert r.status_code == 400
    assert "NEBIUS_API_KEY" in r.json()["error"] and ".env" in r.json()["error"]


def test_run_rejects_bad_mode(monkeypatch):
    c = client(monkeypatch)
    assert c.post("/api/run", json={"mode": "hack", "input": "What is X?"}).status_code == 422


def test_replay_streams_events_then_result(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNS", tmp_path)  # no saved runs -> bundled sample
    c = client(monkeypatch)
    lines = [json.loads(line) for line in c.get("/api/replay?speed=1000").text.splitlines() if line]
    assert lines[0] == {"replay": "sample_run.json"}
    assert any("event" in m for m in lines)
    assert lines[-1]["result"]["status"] == "answered"


def test_index_served(monkeypatch):
    c = client(monkeypatch)
    r = c.get("/")
    assert r.status_code == 200 and "Receipts" in r.text
    assert c.get("/static/app.js").status_code == 200
