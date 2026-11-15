from flask import Blueprint, render_template, request

from app.predictor import predict
from model.schema import CATEGORIES

bp = Blueprint("pages", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["GET", "POST"])
def predict_page():
    result, error = None, None
    if request.method == "POST":
        try:
            result = predict(request.form.to_dict())
        except (KeyError, ValueError) as exc:
            error = f"Invalid input: {exc}"
    return render_template("predict.html", categories=CATEGORIES,
                           result=result, error=error, form=request.form)
