"""Shared feature engineering and preprocessing for the training script and the Streamlit app."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SUB_GRADES = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
SUB_GRADE_MAP = {s: i for i, s in enumerate(SUB_GRADES, start=1)}
CAT_COLS = ["home_ownership", "verification_status", "purpose"]


def build_features(data):
    """Build application-time features from raw loan columns."""
    x = pd.DataFrame(index=data.index)
    issue = pd.to_datetime(data["issue_d"], format="%b-%y")
    earliest = pd.to_datetime(data["earliest_cr_line"], format="%b-%y", errors="coerce")
    wrong_century = earliest > issue
    earliest = earliest.where(~wrong_century, earliest - pd.DateOffset(years=100))
    x["credit_history_years"] = (issue - earliest).dt.days / 365.25

    emp = data["emp_length"].str.replace("< 1 year", "0", regex=False).str.extract(r"(\d+)")[0].astype(float)
    x["emp_years"] = emp
    x["emp_length_missing"] = emp.isna().astype(int)

    x["term_60"] = (data["term"].str.strip() == "60 months").astype(int)
    x["sub_grade_num"] = data["sub_grade"].map(SUB_GRADE_MAP).astype(float)
    x["loan_amnt"] = data["loan_amnt"]
    x["dti"] = data["dti"]
    x["revol_util"] = data["revol_util"]
    for c in ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc",
              "collections_12_mths_ex_med", "acc_now_delinq"]:
        x[c] = data[c]
    for c in ["annual_inc", "revol_bal", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"]:
        x[f"log_{c}"] = np.log1p(data[c].clip(lower=0))

    income = data["annual_inc"].replace(0, np.nan)
    x["loan_to_income"] = data["loan_amnt"] / income
    x["installment_to_income"] = data["installment"] * 12 / income
    x["revol_bal_to_income"] = data["revol_bal"] / income

    x["had_delinq"] = data["mths_since_last_delinq"].notna().astype(int)
    x["mths_since_last_delinq"] = data["mths_since_last_delinq"]
    x["had_public_record"] = data["mths_since_last_record"].notna().astype(int)
    x["mths_since_last_record"] = data["mths_since_last_record"]

    x["home_ownership"] = data["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
    x["verification_status"] = data["verification_status"]
    x["purpose"] = data["purpose"]
    return x


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to percentiles learned from the training data."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def make_preprocessor(columns):
    num_cols = [c for c in columns if c not in CAT_COLS]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("clip", Winsorizer())])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=200), CAT_COLS),
    ])


def recommend(score, review_cutoff, reject_cutoff):
    if score < review_cutoff:
        return "Fast-track"
    if score < reject_cutoff:
        return "Manual review"
    return "Reject or require collateral"
