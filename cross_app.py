import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

# Create a Streamlit interface for file upload and variable selection
st.title("Categorical Variable Analysis")

st.write("This app allows you to perform a preliminary analysis of categorical variables in a CSV file. "
         "First, upload a CSV file that contains categorical variables. The app will automatically detect the categorical variables and allow you to select which ones you want to analyze. "
         "The app will then perform pairwise cross-tabulations of the selected variables and display the contingency tables and corresponding test results. "
         "This analysis is useful for exploring the relationships between pairs of variables, but additional analysis may be required to fully understand the relationships between the variables.")

st.write("When you use this Streamlit app to upload your CSV file, the data from the file will be temporarily stored on the server for the duration of your session. "
         "Once you close the app or the server is restarted, the data will be deleted and will no longer be stored on the server. "
         "It's important to keep in mind that if you are using a public cloud platform, some data may be stored temporarily and not automatically deleted. "
         "To ensure the privacy and security of your data, we recommend using a private cloud or on-premise infrastructure and following relevant privacy regulations.")

file = st.file_uploader("Upload a CSV file")
if file is not None:
    data = pd.read_csv(file)
    categorical_cols = list(data.select_dtypes(include=["object"]).columns)
    selected_cols = st.multiselect("Select categorical variables to analyze", categorical_cols)

    # Convert selected columns to category data type if necessary
    for col in selected_cols:
        if not pd.api.types.is_categorical_dtype(data[col]):
            data[col] = data[col].astype("category")

   # Perform pairwise cross-tabulations and display results
    st.header("Cross-tabulations")
    tabs_list = []
    significant_tabs = []
    for i, col1 in enumerate(selected_cols):
        for col2 in selected_cols[i+1:]:
            crosstab = pd.crosstab(data[col1], data[col2])
            tabs_list.append((col1, col2))
            st.write(f"<a id='{col1}{col2}'></a>", unsafe_allow_html=True)
            st.write(f"Cross-tabulation of {col1} and {col2}")
            if crosstab.shape == (2, 2) and not np.any(crosstab < 5):
                oddsratio, pval = fisher_exact(crosstab)
                st.write(f"Fisher's exact test statistic: {round(oddsratio, 4)}, p-value: {round(pval, 4)}")
                if pval < 0.05:
                    significant_tabs.append((col1, col2))
            else:
                chi2, pval, dof, exp_freq = chi2_contingency(crosstab)
                st.write(f"Chi-square test statistic: {round(chi2, 4)}, p-value: {round(pval, 4)}")
                if pval < 0.05:
                    significant_tabs.append((col1, col2))

    # Display cross-tabs and significant cross-tabs in the sidebar
    tabs_container = st.sidebar.container()
    tabs_container.write("## Cross-tabs")
    with tabs_container:
        tabs_list_elem = []
        for idx, (col1, col2) in enumerate(tabs_list):
            if (col1, col2) in significant_tabs or (col2, col1) in significant_tabs:
                tag = "**(significant)**"
            else:
                tag = ""
            tabs_list_elem.append(f"{idx + 1}. [{col1} x {col2}]{tag}")
        st.sidebar.write("\n".join(tabs_list_elem))
