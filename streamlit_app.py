import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle

st.write("""
# Bank Churner Prediction App


This app predicts which bank customers will churn so that the bank manager can proactively go to 
those customers to provide better serve and potentially turn customers' decision into the opposite direction.
The model deloyed is **gradient boosting**.

Data obtained from (https://www.kaggle.com/sakshigoyal7/credit-card-customers) 
""")

st.sidebar.header('User Input Features')

def user_input_features():
        Gender = st.sidebar.selectbox('Gender',('M','F'))
        Education_Level = st.sidebar.selectbox('Education Level',('College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate',
                                                'Uneducated', 'Unknown'))
        Marital_Status = st.sidebar.selectbox('Martal Status',('Divorced', 'Married', 'Single', 'Unknown'))
        Income_Category = st.sidebar.selectbox('Income Category',('$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K',
                                                'Less than $40K', 'Unknown'))
        Card_Category = st.sidebar.selectbox('Card Category',('Blue', 'Gold', 'Platinum', 'Silver'))
        Months_on_book = st.sidebar.slider('Months on book', 13,56,20)
        Total_Amt_Chng_Q4_Q1 = st.sidebar.slider('Total_Amt_Chng_Q4_Q1', 0.0, 3.397, 2.0)
        Contacts_Count_12_mon = st.sidebar.slider('Contacts_Count_12_mon', 0, 6, 3)
        Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total_Ct_Chng_Q4_Q1', 0.0,3.714, 2.0)
        Months_Inactive_12_mon = st.sidebar.slider('Months_Inactive_12_mon', 0,6,4)
        Total_Revolving_Bal =st.sidebar.slider('Total_Revolving_Bal', 0, 2517, 1000)
        Avg_Utilization_Ratio = st.sidebar.slider('Avg_Utilization_Ratio', 0.0,0.99,0.5)
        Total_Relationship_Count = st.sidebar.slider('Total_Relationship_Count', 1,6,3)
        Credit_Limit = st.sidebar.slider('Credit_Limit', 1438, 34516,2000)
        Total_Trans_Amt = st.sidebar.slider('Total_Trans_Amt', 510,18484, 2000)
        Avg_Open_To_Buy = st.sidebar.slider('Avg_Open_To_Buy', 3,34516, 2000)
        Total_Trans_Ct = st.sidebar.slider('Total_Trans_Ct', 10,139,50)
        Dependent_count = st.sidebar.slider('Dependent_count', 0,5,3)
        Customer_Age = st.sidebar.slider('Customer_Age', 26, 73,30)
        data = {'Gender': Gender,
                'Education_Level': Education_Level,
                'Marital_Status': Marital_Status,
                'Income_Category': Income_Category,
                'Card_Category': Card_Category,
                'Months_on_book': Months_on_book,
                'Total_Amt_Chng_Q4_Q1':Total_Amt_Chng_Q4_Q1,
                'Contacts_Count_12_mon':Contacts_Count_12_mon,
                'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1,
                'Months_Inactive_12_mon':Months_Inactive_12_mon,
                'Total_Revolving_Bal':Total_Revolving_Bal,
                'Avg_Utilization_Ratio':Avg_Utilization_Ratio,
                'Total_Relationship_Count':Total_Relationship_Count,
                'Credit_Limit':Credit_Limit,
                'Total_Trans_Amt':Total_Trans_Amt,
                'Avg_Open_To_Buy':Avg_Open_To_Buy,
                'Total_Trans_Ct':Total_Trans_Ct,
                'Dependent_count':Dependent_count,
                'Customer_Age':Customer_Age
                }
        features = pd.DataFrame(data, index=[0])
        return features

input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
churn_raw = pd.read_csv('data/cleaned_churn.csv')
churn = churn_raw.drop(columns=['churn','Unnamed: 0'])
df = pd.concat([input_df,churn],axis=0)

encode = ['Gender', 'Education_Level',
        'Marital_Status', 'Income_Category', 'Card_Category']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

# Displays the user input features
st.subheader('User Input features')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write("""
        class 0 means not churn, class 1 means churn
        """)

churn_decision = np.array(["Not Churn (class 0)",'Churn (class 1)'])
st.write(churn_decision[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


# use this to run in command line : streamlit run streamlit_app.py
# acknowledgement: https://www.youtube.com/watch?v=ZZ4B0QUHuNc