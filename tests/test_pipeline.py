import pandas as pd

from retailmind.pipeline import build_pipeline


def test_pipeline_fits_and_predicts_with_unseen_category():

    X = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48, 60],
        "monthly": [90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
        "contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
            "Two year"
        ]
    })

    y = pd.Series([1, 1, 0, 1, 0, 0])

    model = build_pipeline(X)

    model.fit(X, y)

    sample = pd.DataFrame({
        "tenure": [8],
        "monthly": [75.0],
        "contract": ["New contract type"]
    })

    predictions = model.predict(sample)

    probabilities = model.predict_proba(sample)

    assert predictions.shape == (1,)
    assert probabilities.shape == (1, 2)