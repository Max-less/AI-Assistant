"""Auth + per-user isolation tests."""


def register(client, email="a@example.com", name="Alice", password="secret123"):
    r = client.post(
        "/api/auth/register",
        json={"email": email, "name": name, "password": password},
    )
    return r


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_returns_token_and_user(client):
    r = register(client)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "a@example.com"
    assert body["user"]["is_guest"] is False
    assert body["user"]["guest_remaining"] is None


def test_register_duplicate_email_rejected(client):
    assert register(client).status_code == 200
    r = register(client, name="Other")
    assert r.status_code == 400


def test_login_success_and_failure(client):
    register(client)
    ok = client.post("/api/auth/login", json={"email": "a@example.com", "password": "secret123"})
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad_pw = client.post("/api/auth/login", json={"email": "a@example.com", "password": "nope"})
    assert bad_pw.status_code == 401

    unknown = client.post("/api/auth/login", json={"email": "x@example.com", "password": "secret123"})
    assert unknown.status_code == 401


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401

    token = register(client).json()["access_token"]
    me = client.get("/api/auth/me", headers=auth_header(token))
    assert me.status_code == 200
    assert me.json()["email"] == "a@example.com"


def test_protected_endpoints_require_auth(client):
    assert client.get("/api/sessions").status_code == 401
    assert client.post("/api/chat", json={"question": "hi", "session_id": None}).status_code == 401


def test_sessions_are_isolated_per_user(client):
    token_a = register(client, email="a@example.com").json()["access_token"]
    token_b = register(client, email="b@example.com").json()["access_token"]

    chat = client.post(
        "/api/chat",
        json={"question": "What is Scrum?", "session_id": None},
        headers=auth_header(token_a),
    )
    assert chat.status_code == 200
    session_id = chat.json()["session_id"]

    # A sees its session; B sees nothing.
    a_sessions = client.get("/api/sessions", headers=auth_header(token_a)).json()
    b_sessions = client.get("/api/sessions", headers=auth_header(token_b)).json()
    assert len(a_sessions) == 1
    assert b_sessions == []

    # B cannot read A's history.
    cross = client.get(f"/api/history/{session_id}", headers=auth_header(token_b))
    assert cross.status_code == 404
    own = client.get(f"/api/history/{session_id}", headers=auth_header(token_a))
    assert own.status_code == 200


def test_guest_query_limit(client):
    guest = client.post("/api/auth/guest")
    assert guest.status_code == 200
    body = guest.json()
    assert body["user"]["is_guest"] is True
    assert body["user"]["guest_remaining"] == 5
    token = body["access_token"]

    session_id = None
    for i in range(5):
        r = client.post(
            "/api/chat",
            json={"question": f"q{i}", "session_id": session_id},
            headers=auth_header(token),
        )
        assert r.status_code == 200, f"query {i} should succeed"
        session_id = r.json()["session_id"]

    # 6th query is blocked.
    blocked = client.post(
        "/api/chat",
        json={"question": "q5", "session_id": session_id},
        headers=auth_header(token),
    )
    assert blocked.status_code == 403

    me = client.get("/api/auth/me", headers=auth_header(token))
    assert me.json()["guest_remaining"] == 0


def test_guest_quota_not_reset_by_relogin(client):
    cid = "browser-abc-123"

    # First guest session for this browser: exhaust the quota.
    token = client.post("/api/auth/guest", json={"client_id": cid}).json()["access_token"]
    session_id = None
    for i in range(5):
        r = client.post(
            "/api/chat",
            json={"question": f"q{i}", "session_id": session_id},
            headers=auth_header(token),
        )
        assert r.status_code == 200
        session_id = r.json()["session_id"]

    # "Log out" and back in as guest with the SAME client_id -> same account,
    # quota stays exhausted.
    again = client.post("/api/auth/guest", json={"client_id": cid}).json()
    assert again["user"]["guest_remaining"] == 0
    blocked = client.post(
        "/api/chat",
        json={"question": "again", "session_id": None},
        headers=auth_header(again["access_token"]),
    )
    assert blocked.status_code == 403

    # A different browser id gets a fresh quota.
    other = client.post("/api/auth/guest", json={"client_id": "browser-xyz-999"}).json()
    assert other["user"]["guest_remaining"] == 5
    assert other["user"]["id"] != again["user"]["id"]


def test_feedback_scoped_to_owner(client):
    token_a = register(client, email="a@example.com").json()["access_token"]
    token_b = register(client, email="b@example.com").json()["access_token"]

    chat = client.post(
        "/api/chat",
        json={"question": "hi", "session_id": None},
        headers=auth_header(token_a),
    )
    message_id = chat.json()["message"]["id"]

    # Owner can leave feedback.
    ok = client.post(
        "/api/feedback",
        json={"message_id": message_id, "value": 1},
        headers=auth_header(token_a),
    )
    assert ok.status_code == 200

    # Another user cannot.
    forbidden = client.post(
        "/api/feedback",
        json={"message_id": message_id, "value": 1},
        headers=auth_header(token_b),
    )
    assert forbidden.status_code == 404
