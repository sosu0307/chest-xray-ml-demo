import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data/processed/ML/test_data")

df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/ML/features_reduced_15.csv"))

st.title("Covid-19 Diagnosis Using Chest X-Rays")
st.sidebar.title("Table of contents")
pages = ["Data Exploration", "Data Visualization", "Baseline Models", "Deep Learning"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("#### X-Ray")
        st.image(os.path.join(CURRENT_DIR, "COVID-32-xray.png"), use_column_width=True)
    with col2:
        st.write("#### Mask")
        st.image(os.path.join(CURRENT_DIR, "COVID-32-mask.png"), use_column_width=True)
    with col3:
        st.write("#### Isolated Lung")
        st.image(
            os.path.join(CURRENT_DIR, "COVID-32-isolated.png"), use_column_width=True
        )

    st.write("###")
    st.write("## Dataframe of extracted features")
    st.write("#### Shape:", df.shape[0], "rows", df.shape[1], "features")
    st.write("#### Head")
    st.dataframe(df.head(10))
    st.write("#### Statistics")
    st.dataframe(df.describe())

if page == pages[1]:
    fig = plt.figure()
    class_count = sns.countplot(
        x="target", data=df, palette={"0": "#1f77b4", "1": "#ff7f0e"}
    )
    class_count.set_xticklabels(["non_covid", "covid"])
    st.pyplot(fig)

if page == pages[2]:
    # Load data
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)

    def prediction(classifier):
        if classifier == "Logistic regression":
            clf = joblib.load(os.path.join(MODELS_DIR, "logistic_none.pkl"))
        elif classifier == "Logistic regression oversample":
            clf = joblib.load(os.path.join(MODELS_DIR, "logistic_oversample.pkl"))
        elif classifier == "Logistic regression smote":
            clf = joblib.load(os.path.join(MODELS_DIR, "logistic_smote.pkl"))
        elif classifier == "Logistic regression undersample":
            clf = joblib.load(os.path.join(MODELS_DIR, "logistic_undersample.pkl"))
        elif classifier == "Random forest":
            clf = joblib.load(os.path.join(MODELS_DIR, "rf_none.pkl"))
        elif classifier == "Random forest oversample":
            clf = joblib.load(os.path.join(MODELS_DIR, "rf_oversample.pkl"))
        elif classifier == "Random forest smote":
            clf = joblib.load(os.path.join(MODELS_DIR, "rf_smote.pkl"))
        elif classifier == "Random forest undersample":
            clf = joblib.load(os.path.join(MODELS_DIR, "rf_undersample.pkl"))
        elif classifier == "SVC":
            clf = joblib.load(os.path.join(MODELS_DIR, "svc_undersample.pkl"))
        return clf

    choice = [
        "Logistic regression",
        "Logistic regression oversample",
        "Logistic regression smote",
        "Logistic regression undersample",
        "Random forest",
        "Random forest oversample",
        "Random forest smote",
        "Random forest undersample",
        "SVC",
    ]
    option = st.selectbox("Choice of the model", choice)
    st.write("The chosen model is:", option)
    clf = prediction(option)
    y_pred = clf.predict(X_test_scaled)
    st.dataframe(
        pd.crosstab(
            y_test.iloc[:, 0], y_pred, rownames=["Real Class"], colnames=["Pred Class"]
        )
    )
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
