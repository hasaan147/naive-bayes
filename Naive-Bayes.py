import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import load_iris

def main():
    st.title('Naive Bayes Classification')
    
    # Load the Iris dataset
    st.subheader('Iris Dataset Overview')
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Select features and target column
    st.sidebar.subheader('Select Features and Target')
    features = st.sidebar.multiselect('Features', iris.feature_names, default=iris.feature_names)
    target = st.sidebar.selectbox('Target', ['target'])
    
    if features and target:
        X = df[features]
        y = df[target]
        
        # Encode labels if they are not numeric
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize the features for GaussianNB
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize and train Gaussian Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred_gnb = gnb.predict(X_test)
        
        # Display results
        st.write("### Gaussian Naive Bayes")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_gnb)}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_gnb))
        
        # Prepare data for Multinomial Naive Bayes
        min_max_scaler = MinMaxScaler()
        X_train_mnb = min_max_scaler.fit_transform(X_train)
        X_test_mnb = min_max_scaler.transform(X_test)
        
        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(X_train_mnb, y_train)
        y_pred_mnb = mnb.predict(X_test_mnb)
        
        st.write("\n### Multinomial Naive Bayes")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_mnb)}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_mnb))
        
        # Prepare data for Bernoulli Naive Bayes
        X_train_bin = (X_train > 0).astype(int)
        X_test_bin = (X_test > 0).astype(int)
        
        bnb = BernoulliNB(alpha=1.0)
        bnb.fit(X_train_bin, y_train)
        y_pred_bnb = bnb.predict(X_test_bin)
        
        st.write("\n### Bernoulli Naive Bayes")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_bnb)}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_bnb))
    else:
        st.write("Please select features and target column.")

if __name__ == "__main__":
    main()
