from flask import Blueprint, jsonify, request

from app.predictor import load_model, predict

bp = Blueprint("api", __name__)


@bp.get("/health")
def health():
    return jsonify(status="ok", model_loaded=load_model() is not None)


@bp.post("/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(predict(payload))
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400
