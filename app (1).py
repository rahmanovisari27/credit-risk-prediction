"""Credit risk scoring demo. Run with: streamlit run app.py"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from credit_risk_lib import SUB_GRADES, build_features, recommend
import ui

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car",
            "medical", "moving", "vacation", "wedding", "house", "educational", "renewable_energy", "other"]
EMP = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years",
       "9 years", "10+ years", "Unknown"]


@st.cache_resource
def load_model():
    with open("model/metadata.json") as f:
        return joblib.load("model/credit_risk_model.joblib"), json.load(f)


model, meta = load_model()

REQUIRED = ["loan_amnt", "term", "sub_grade", "installment", "annual_inc", "dti",
            "home_ownership", "verification_status", "purpose"]
OPTIONAL_NUMERIC = ["revol_bal", "revol_util", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc",
                    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
                    "mths_since_last_delinq", "mths_since_last_record"]
OPTIONAL_TEXT = ["emp_length", "earliest_cr_line", "issue_d"]


def score_frame(data):
    """Clean raw loan columns and return probability of default, risk band and action per row."""
    data = data.copy()
    for col in OPTIONAL_NUMERIC + ["loan_amnt", "installment", "annual_inc", "dti"]:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
    for col in OPTIONAL_TEXT:
        if col not in data:
            data[col] = None
        data[col] = data[col].astype("object").where(data[col].notna(), None)
    issue = pd.to_datetime(data["issue_d"], format="%b-%y", errors="coerce")
    data.loc[issue.isna(), "issue_d"] = date.today().strftime("%b-%y")
    for col in ["term", "sub_grade", "home_ownership", "verification_status", "purpose"]:
        data[col] = data[col].astype(str).str.strip()
    data["emp_length"] = data["emp_length"].astype("object")

    scores = model.predict_proba(build_features(data)[meta["features"]])[:, 1]
    bands = np.searchsorted(meta["band_edges"], scores, side="right") + 1
    return pd.DataFrame({
        "probability_of_default": scores.round(4),
        "risk_band": bands,
        "recommended_action": [recommend(v, meta["review_cutoff"], meta["reject_cutoff"]) for v in scores],
    }, index=data.index)


st.markdown(ui.hero("Credit Risk Scoring",
                    "Estimate the probability of default using only information available at application. "
                    "Portfolio demo trained on loans from 2007 to 2013, not validated on Indonesian data, "
                    "and not for real credit decisions.",
                    "ID/X Partners x Rakamin Academy · Rahma Novita"), unsafe_allow_html=True)

single_tab, batch_tab = st.tabs(["Single application", "Batch from CSV"])

with single_tab:


    with st.form("application"):
        loan_col, borrower_col, credit_col = st.columns(3)
        with loan_col:
            st.subheader("Loan")
            loan_amnt = st.number_input("Loan amount ($)", 500, 35000, 15000, step=500)
            installment = st.number_input("Monthly installment ($)", 15, 1500, 510)
            term = st.selectbox("Term", ["36 months", "60 months"])
            purpose = st.selectbox("Purpose", PURPOSES)
            sub_grade = st.selectbox("Sub-grade", SUB_GRADES, index=SUB_GRADES.index("C3"))
        with borrower_col:
            st.subheader("Borrower")
            annual_inc = st.number_input("Annual income ($)", 1000, 1000000, 60000, step=1000)
            emp_length = st.selectbox("Employment length", EMP, index=4)
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            verification_status = st.selectbox("Income verification", ["Not Verified", "Source Verified", "Verified"])
            dti = st.number_input("Debt-to-income (%)", 0.0, 40.0, 18.0)
        with credit_col:
            st.subheader("Credit history")
            history_years = st.number_input("Length of credit history (years)", 1, 60, 10)
            open_acc = st.number_input("Open credit lines", 0, 80, 11)
            total_acc = st.number_input("Total credit lines", 1, 150, 22)
            revol_bal = st.number_input("Revolving balance ($)", 0, 500000, 12000, step=500)
            revol_util = st.number_input("Revolving utilization (%)", 0.0, 150.0, 60.0)
            inq_last_6mths = st.number_input("Credit inquiries, last 6 months", 0, 30, 1)

        with st.expander("More credit details (blank fields use the training median)"):
            more_a, more_b = st.columns(2)
            with more_a:
                delinq_2yrs = st.number_input("Delinquencies, last 2 years", 0, 30, 0)
                pub_rec = st.number_input("Public records", 0, 20, 0)
                mths_delinq = st.number_input("Months since last delinquency", 0, 200, None, placeholder="Never")
                mths_record = st.number_input("Months since last public record", 0, 200, None, placeholder="Never")
            with more_b:
                tot_cur_bal = st.number_input("Total current balance ($)", 0, 9000000, None, placeholder="Unknown")
                total_rev_hi_lim = st.number_input("Total revolving limit ($)", 0, 3000000, None, placeholder="Unknown")
                tot_coll_amt = st.number_input("Total collection amount ($)", 0, 10000000, None, placeholder="Unknown")

        submitted = st.form_submit_button("Score application", type="primary")

    if submitted:
        today = date.today()
        application = pd.DataFrame([{
            "issue_d": today.strftime("%b-%y"),
            "earliest_cr_line": today.replace(day=1, year=today.year - history_years).strftime("%b-%y"),
            "emp_length": None if emp_length == "Unknown" else emp_length,
            "term": term, "sub_grade": sub_grade, "loan_amnt": loan_amnt, "installment": installment,
            "annual_inc": annual_inc, "dti": dti, "revol_bal": revol_bal, "revol_util": revol_util,
            "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
            "total_acc": total_acc, "collections_12_mths_ex_med": 0, "acc_now_delinq": 0,
            "tot_coll_amt": tot_coll_amt, "tot_cur_bal": tot_cur_bal, "total_rev_hi_lim": total_rev_hi_lim,
            "mths_since_last_delinq": mths_delinq, "mths_since_last_record": mths_record,
            "home_ownership": home_ownership, "verification_status": verification_status, "purpose": purpose,
        }])
        for col in ["tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "mths_since_last_delinq", "mths_since_last_record"]:
            application[col] = pd.to_numeric(application[col], errors="coerce")
        application["emp_length"] = application["emp_length"].astype("object")

        score = float(model.predict_proba(build_features(application)[meta["features"]])[0, 1])
        band = int(np.searchsorted(meta["band_edges"], score, side="right")) + 1
        action = recommend(score, meta["review_cutoff"], meta["reject_cutoff"])

        st.write("")
        card_col, guide_col = st.columns([3, 2], gap="large")
        with card_col:
            st.markdown(ui.insight_card(score, band, action, meta["band_bad_rate"][str(band)]), unsafe_allow_html=True)
        with guide_col:
            st.markdown(ui.decision_guide(score, meta["review_cutoff"], meta["reject_cutoff"], action),
                        unsafe_allow_html=True)


with batch_tab:
    st.write("Upload a CSV with one application per row. Column names follow the original loan dataset.")
    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in REQUIRED))
    st.caption("Optional columns: " + ", ".join(OPTIONAL_NUMERIC + OPTIONAL_TEXT) + ". Missing optional values use "
               "the training median. If `loan_status` is present, the actual outcome is shown for comparison.")
    try:
        with open("sample_applications.csv", "rb") as f:
            st.download_button("Download sample CSV", f, "sample_applications.csv", "text/csv")
    except FileNotFoundError:
        pass

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
        except Exception as err:
            st.error(f"The file could not be read as CSV: {err}")
            st.stop()
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()
        bad_grade = ~raw["sub_grade"].astype(str).str.strip().isin(SUB_GRADES)
        if bad_grade.any():
            st.warning(f"{int(bad_grade.sum())} rows have an unknown sub_grade and were skipped.")
        raw = raw[~bad_grade]
        if raw.empty:
            st.stop()

        result = score_frame(raw)
        output = pd.concat([raw.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
        if "loan_status" in raw.columns:
            bad = ["Charged Off", "Default", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off"]
            good = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
            status = raw["loan_status"].reset_index(drop=True)
            output["actual_outcome"] = np.where(status.isin(bad), "BAD", np.where(status.isin(good), "GOOD", "Unknown"))

        st.success(f"Scored {len(output):,} applications.")
        scores = output["probability_of_default"].to_numpy()
        actions = output["recommended_action"]
        has_actual = "actual_outcome" in output and (output["actual_outcome"] != "Unknown").any()
        st.markdown(ui.hero("Portfolio Score Dashboard",
                            "Risk overview of the uploaded applications, scored by the credit risk model.",
                            "Batch scoring"), unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        fast, review, reject = [(actions == a).mean() for a in ui.ACTION_COLOR]
        k1.markdown(ui.kpi("Applications", f"{len(output):,}", f"Median PD {np.median(scores):.1%}",
                           f"Max {scores.max():.1%}", 1.0), unsafe_allow_html=True)
        k2.markdown(ui.kpi("Average probability of default", f"{scores.mean():.1%}", "Model estimate",
                           f"AUC {meta['auc_test']:.3f}", scores.mean()), unsafe_allow_html=True)
        k3.markdown(ui.kpi("Fast-track share", f"{fast:.1%}", f"Review {review:.0%}", "of applications", fast),
                    unsafe_allow_html=True)
        if has_actual:
            known = output[output["actual_outcome"] != "Unknown"]
            k4.markdown(ui.kpi("Actual bad rate", f"{(known['actual_outcome'] == 'BAD').mean():.1%}",
                               f"{len(known):,} with outcome", "from loan_status",
                               (known['actual_outcome'] == 'BAD').mean()), unsafe_allow_html=True)
        else:
            k4.markdown(ui.kpi("Reject share", f"{reject:.1%}", "Reject or collateral", "of applications", reject),
                        unsafe_allow_html=True)

        st.write("")
        c1, c2, c3 = st.columns([5, 4, 3], gap="medium")
        c1.markdown(ui.histogram(scores, meta["review_cutoff"], meta["reject_cutoff"]), unsafe_allow_html=True)
        if has_actual:
            known = output[output["actual_outcome"] != "Unknown"]
            c2.markdown(ui.band_chart(known["risk_band"].to_numpy(),
                                      (known["actual_outcome"] == "BAD").to_numpy()), unsafe_allow_html=True)
        else:
            c2.markdown(ui.band_chart(output["risk_band"].to_numpy()), unsafe_allow_html=True)
        c3.markdown(ui.action_breakdown(actions), unsafe_allow_html=True)

        if has_actual:
            known = output[output["actual_outcome"] != "Unknown"]
            summary = (known.assign(bad=known["actual_outcome"].eq("BAD"))
                       .groupby("recommended_action")["bad"].agg(["size", "mean"])
                       .reindex(list(ui.ACTION_COLOR)).dropna()
                       .rename(columns={"size": "Loans", "mean": "Actual bad rate"}))
            st.write("")
            st.markdown("#### Actual outcome by recommended action")
            st.dataframe(summary, column_config={
                "Loans": st.column_config.NumberColumn(format="%d"),
                "Actual bad rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

        st.markdown("#### Scored applications")
        show = [c for c in ["id", "loan_amnt", "term", "sub_grade", "annual_inc", "purpose",
                            "probability_of_default", "risk_band", "recommended_action", "actual_outcome"] if c in output]
        st.dataframe(output[show], width="stretch", hide_index=True, column_config={
            "probability_of_default": st.column_config.ProgressColumn(
                "Probability of default", format="percent", min_value=0, max_value=1),
            "risk_band": st.column_config.NumberColumn("Risk band", format="%d")})
        st.download_button("Download results", output.to_csv(index=False).encode("utf-8"),
                           "credit_risk_predictions.csv", "text/csv")
