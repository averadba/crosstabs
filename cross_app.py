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
    for i, col1 in enumerate(selected_cols):
        for col2 in selected_cols[i+1:]:
            crosstab = pd.crosstab(data[col1], data[col2])
            exp_freq = chi2_contingency(crosstab)[3]
            if crosstab.shape == exp_freq.shape:
                st.write(f"<a id='{col1}{col2}'></a>", unsafe_allow_html=True)
                st.write(f"Cross-tabulation of {col1} and {col2}")
                st.write(pd.concat([crosstab, pd.DataFrame(exp_freq, index=crosstab.index, columns=crosstab.columns, prefix='Expected_')], axis=1))
            else:
                raise ValueError("Dimensions of crosstab and exp_freq do not match")
            chi2, pval, dof, _ = chi2_contingency(crosstab)
            if crosstab.shape == (2, 2):
                oddsratio, pval = fisher_exact(crosstab)
                st.write(f"Fisher's exact test statistic: {oddsratio}, p-value: {pval}")
            elif np.any(exp_freq < 5):
                st.write("<p style='color: red;'>Warning:</p> chi-square test may be invalid due to expected frequency less than 5", unsafe_allow_html=True)
                st.write(f"Chi-square test statistic: {chi2}, p-value: {pval}")
            else:
                st.write(f"Chi-square test statistic: {chi2}, p-value: {pval}")
            tabs_list.append((col1, col2))

    st.sidebar.header("Cross-tabulations")
    for tab in tabs_list:
        st.sidebar.write(f"{tab[0]} x {tab[1]}")
