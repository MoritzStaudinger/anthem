def test_recent_caches_returns_list(flask_client):
    resp = flask_client.get("/api/recent_caches")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_feedback_validates_required_fields(flask_client):
    resp = flask_client.post("/api/feedback", json={})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Missing required fields"}


def test_generate_query_validates_required_fields(flask_client):
    resp = flask_client.post("/api/generate_query", json={})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Missing required fields"}


def test_generate_query_only_accepts_post(flask_client):
    resp = flask_client.get("/api/generate_query")
    assert resp.status_code == 405


def test_unknown_route_returns_404(flask_client):
    resp = flask_client.get("/api/nope")
    assert resp.status_code == 404


def test_feedback_returns_404_for_unknown_cache(flask_client):
    resp = flask_client.post(
        "/api/feedback",
        json={"prompt": "p", "model": "gpt-4o-mini", "rating": 5, "comments": "ok"},
    )
    assert resp.status_code == 404
    assert "No cached entry found" in resp.get_json().get("error", "")
