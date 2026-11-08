"""Train the selected model and save it for the Streamlit app.

Usage: python train_model.py path/to/loan_data_2007_2014.csv
Run this again if the saved model cannot be loaded in your environment.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from credit_risk_lib import build_features, make_preprocessor

BAD = ["Charged Off", "Default", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off"]
GOOD = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
SEED = 42

path = sys.argv[1] if len(sys.argv) > 1 else "loan_data_2007_2014.csv"
raw = pd.read_csv(path, low_memory=False)
data = raw[raw["loan_status"].isin(BAD + GOOD)].copy()
y = data["loan_status"].isin(BAD).astype(int)
issue = pd.to_datetime(data["issue_d"], format="%b-%y")
X = build_features(data)
features = list(X.columns)
is_train = issue <= "2013-12-31"
X_train, X_test, y_train, y_test = X[is_train], X[~is_train], y[is_train], y[~is_train]

# Best parameters from the randomized search in credit_risk_prediction.ipynb
model = Pipeline([("prep", make_preprocessor(features)),
                  ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                                         min_samples_leaf=500, l2_regularization=10.0,
                                                         random_state=SEED))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
model.fit(X_train, y_train)
test_scores = model.predict_proba(X_test)[:, 1]

fpr, tpr, thr = roc_curve(y_train, oof)
review_cutoff = float(thr[int(np.argmax(tpr - fpr))])
net = (data["total_pymnt"] - data["funded_amnt"])[is_train].values
candidates = np.quantile(oof, np.linspace(0.3, 1, 71))
candidates[-1] = 1.01
reject_cutoff = float(candidates[int(np.argmax([net[oof < c].sum() for c in candidates]))])

edges = np.quantile(oof, np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
bands = pd.cut(test_scores, edges, labels=range(1, 11))
band_bad = pd.Series(y_test.values).groupby(np.asarray(bands)).mean()

meta = {"features": features, "review_cutoff": review_cutoff, "reject_cutoff": reject_cutoff,
        "band_edges": [float(v) for v in edges[1:-1]],
        "band_bad_rate": {str(int(k)): float(v) for k, v in band_bad.items()},
        "auc_test": float(roc_auc_score(y_test, test_scores))}
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/credit_risk_model.joblib", compress=3)
with open("model/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=2))
