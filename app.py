import os
import pandas as pd
import streamlit as st
import time  # Required for simulating incremental progress in the progress bar

from pycaret.regression import setup, compare_models, pull, save_model
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# Set the page configuration as the first command immediately after imports
st.set_page_config(page_title="DataWizPro", layout="wide")

def main():
    # CSS for custom styles
    st.markdown("""
        <style>
        .reportview-container .main .block-container {padding-top: 2rem;}
        .css-18e3th9 {padding-top: 3rem;}
        .css-1d391kg {padding: 1rem;}
        .css-1aumxhk {background-color: #121212; color: #ffffff;}
        .css-1d391kg {background-color: #333333; color: #ffffff;}
        .css-2trqyj {background-color: #6200ee;}
        </style>
        """, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        st.image("https://t4.ftcdn.net/jpg/05/74/16/75/360_F_574167578_xZDF0JwUFieaZ8LwbVgUbWZsID7IVFal.jpg", width=300)
        st.title(" Your AI-powered Data Assistant")
        selected_page = st.radio("Menu", ["Upload Data", "Data Profiling", "Model Training", "Download Model"], key="sidebar_menu")
        st.session_state['current_page'] = selected_page

    # Page content based on session state
    if st.session_state['current_page'] == "Upload Data":
        st.header("Data Upload")
        file = st.file_uploader("Choose a csv format file")
        if file:
            df = pd.read_csv(file)
            df.to_csv('dataset.csv', index=False)
            st.success("Dataset successfully uploaded and displayed below:")
            st.dataframe(df)

    elif st.session_state['current_page'] == "Data Profiling":
        st.header("Exploratory Data Analysis")
        df = pd.read_csv('dataset.csv')
        profile_report = ProfileReport(df, explorative=True)
        st_profile_report(profile_report)

    elif st.session_state['current_page'] == "Model Training":
        st.header("Model Training")
        df = pd.read_csv('dataset.csv')
        target_column = st.selectbox('Select the Target Variable', df.columns)
        train_button = st.button('Train Models', key="TrainModels")
        if train_button:
            with st.spinner('Models are being trained...'):
                progress_bar = st.progress(0)
                s = setup(data=df, target=target_column, verbose=False)
                for i in range(1, 101):
                    time.sleep(0.1)  # Simulate time delay for progress
                    progress_bar.progress(i)
                best_model = compare_models()
                st.write("Comparison of Models:")
                st.dataframe(pull())
                save_model(best_model, 'best_model.pkl')
                st.success("Best Model trained and successfully saved.")
                progress_bar.empty()

    elif st.session_state['current_page'] == "Download Model":
        st.header("Download Model")
        try:
            with open('best_model.pkl', 'rb') as f:
                st.download_button("Download The Best Model", f, "best_model.pkl")
        except FileNotFoundError:
            st.error("No trained model found. Please train a model in the 'Model Training' section first.")

if __name__ == "__main__":
    main()
