# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")  # allows override via env var

# ---------- Helpers ----------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def build_feature_df(user_inputs: dict) -> pd.DataFrame:
    """
    Convert raw widget inputs into the dataframe your model expects.
    Edit this to match your training pipeline (column order, types, encodings).
    """
    # Example expected columns (EDIT THESE)
    cols = [
        "age",              # numeric
        "income",           # numeric
        "owns_house",       # binary 0/1
        "city",             # categorical (one of: 'A','B','C')
    ]

    # Basic one-hot encoding for 'city' as an example
    city_categories = ["A", "B", "C"]  # EDIT to match your training
    city_ohe = {f"city_{c}": 0 for c in city_categories}
    if user_inputs["city"] in city_categories:
        city_ohe[f"city_{user_inputs['city']}"] = 1

    base = {
        "age": float(user_inputs["age"]),
        "income": float(user_inputs["income"]),
        "owns_house": int(user_inputs["owns_house"]),
    }

    # Combine into one row; order columns to match training if needed
    row = {**base, **city_ohe}
    df = pd.DataFrame([row])

    # If your model expects a specific column order, enforce it here:
    # df = df[["age","income","owns_house","city_A","city_B","city_C"]]
    return df

def safe_predict(model, X: pd.DataFrame):
    """
    Handles regressors/classifiers gracefully.
    Returns: dict with keys: kind ('regression'|'classification'),
             prediction (np.ndarray), proba (np.ndarray|None), labels (list|None)
    """
    out = {"kind": None, "prediction": None, "proba": None, "labels": None}
    # Try to detect classifier
    has_predict_proba = hasattr(model, "predict_proba")
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    out["prediction"] = np.array(y_pred)

    if has_predict_proba:
        out["kind"] = "classification"
        try:
            proba = model.predict_proba(X)
            out["proba"] = np.array(proba)
        except Exception:
            out["proba"] = None
        # Class labels if available
        classes_ = getattr(model, "classes_", None)
        out["labels"] = classes_.tolist() if classes_ is not None else None
    else:
        out["kind"] = "regression"

    return out

# ---------- UI ----------
st.set_page_config(page_title="PKL Model Demo", page_icon="🤖", layout="centered")

st.title("PKL Model Prediction App")

with st.sidebar:
    st.header("Model")
    st.caption(f"Loading model from: {MODEL_PATH}")
    load_button = st.button("Reload model")

try:
    model = load_model()
    if load_button:
        # Streamlit cache bust: clear and reload
        load_model.clear()
        model = load_model()
        st.sidebar.success("Model reloaded.")
except FileNotFoundError:
    st.error(f"Model file not found at '{MODEL_PATH}'. Place your .pkl in the repo or set MODEL_PATH.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.subheader("Enter input features")

# --------- EDIT THESE INPUTS to match your model ---------
age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
income = st.number_input("Annual Income", min_value=0.0, max_value=1_000_000.0, value=60_000.0, step=1_000.0)
owns_house = st.selectbox("Owns House", options=[0, 1], index=1)
city = st.selectbox("City", options=["A", "B", "C"], index=0)
# ----------------------------------------------------------

inputs = {
    "age": age,
    "income": income,
    "owns_house": owns_house,
    "city": city,
}

with st.expander("Preview feature row"):
    st.write(build_feature_df(inputs))

if st.button("Predict"):
    X = build_feature_df(inputs)
    try:
        result = safe_predict(model, X)
        if result["kind"] == "classification":
            pred_label = result["prediction"][0]
            st.success(f"Predicted class: {pred_label}")
            if result["proba"] is not None and result["labels"] is not None:
                proba_row = result["proba"][0]
                proba_df = pd.DataFrame({"class": result["labels"], "probability": proba_row})
                proba_df["probability"] = proba_df["probability"].round(4)
                st.subheader("Class probabilities")
                st.dataframe(proba_df, use_container_width=True)
        else:
            pred_value = float(result["prediction"][0])
            st.success(f"Predicted value: {pred_value:.4f}")
    except Exception as e:
        st.error(str(e))

st.caption("Tip: Adjust the feature builder to mirror your training pipeline (encoders, scalers, column order).")
