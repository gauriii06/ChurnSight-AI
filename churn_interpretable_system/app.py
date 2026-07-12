from preprocessing.preprocessing import X_train
from flask import Flask, request, render_template
import pandas as pd

from models.random_forest_model import model
from decision_support.risk_scoring import classify_risk
from decision_support.recommendation_engine import generate_recommendations

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():

    probability = None
    risk = None
    recommendations = None

    if request.method == "POST":

        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        contract = request.form["contract"]

        # Create empty dataframe with training columns
        df = pd.DataFrame(columns=X_train.columns)
        df.loc[0] = 0

        # Fill known values
        df.loc[0, "tenure"] = tenure
        df.loc[0, "MonthlyCharges"] = monthly_charges

        if contract == "month":
            df.loc[0, "Contract_Month-to-month"] = 1
        elif contract == "one":
            df.loc[0, "Contract_One year"] = 1
        elif contract == "two":
            df.loc[0, "Contract_Two year"] = 1

        prob = model.predict_proba(df)[0][1]

        probability = round(prob, 3)

        risk = classify_risk(prob)

        active_features = [col for col in df.columns if df.loc[0, col] == 1]
        recommendations = generate_recommendations(active_features)

    return render_template(
        "index.html",
        probability=probability,
        risk=risk,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)