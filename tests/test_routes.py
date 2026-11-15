import pytest

from app import create_app

FORM = {
    "amt": "120.50", "trans_datetime": "2020-07-01T23:15", "category": "shopping_net",
    "merch_lat": "35.47", "merch_long": "-97.52", "dob": "1985-03-02", "gender": "F",
    "lat": "35.22", "long": "-97.44", "city_pop": "128000",
}


@pytest.fixture
def client():
    return create_app().test_client()


def test_pages_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/predict").status_code == 200


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_predict_form(client):
    resp = client.post("/predict", data=FORM)
    assert resp.status_code == 200
    assert b"Engineered features" in resp.data


def test_predict_api_builds_features(client):
    body = client.post("/api/predict", json=FORM).get_json()
    assert body["features"]["is_night"] == 1
    assert body["features"]["hour"] == 23


def test_predict_api_rejects_bad_input(client):
    assert client.post("/api/predict", json={"amt": "5"}).status_code == 400
