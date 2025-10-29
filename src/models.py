# File: src/models.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import custom utility
from src.utils import map_brand_to_tier


# --- RECOMMENDATION MODEL SETUP ---
@st.cache_resource
def setup_recommendation_model(df_rec):
    """Sets up the TF-IDF vectorizer and transforms features."""
    features_in_data = ['Make', 'Model', 'Year', 'Fuel Type', 'Transmission', 'Engine', 'Max Power']

    df_rec_copy = df_rec[features_in_data].fillna('')
    for feature in features_in_data:
        df_rec_copy[feature] = df_rec_copy[feature].astype(str)

    combined_features = df_rec_copy.apply(lambda row: ' '.join(row[f] for f in features_in_data), axis=1)

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    return vectorizer, feature_vectors


# --- PRICE PREDICTION MODEL SETUP ---
@st.cache_resource
def setup_price_prediction_model(df_price):
    """Sets up and fits the price prediction pipeline."""
    df_price_copy = df_price.copy()

    # Feature engineering
    df_price_copy['Car Age'] = df_price_copy['Vehicle Age']
    df_price_copy['log_Price'] = np.log1p(df_price_copy['Price'])
    df_price_copy['Brand_Tier'] = df_price_copy['Make'].apply(map_brand_to_tier)

    numeric_features = ['Kilometer', 'Car Age', 'Engine', 'Max Power', 'Mileage', 'Seats']

    # --- Make replaced by Brand_Tier ---
    categorical_features = ['Model', 'Fuel Type', 'Transmission', 'Seller Type', 'Brand_Tier']

    # Drop missing values
    df_price_copy.dropna(subset=['log_Price'], inplace=True)
    df_price_copy.dropna(subset=numeric_features + categorical_features, inplace=True)

    # Separate features and target
    X = df_price_copy[numeric_features + categorical_features].copy()
    y = df_price_copy['log_Price']

    # Convert categorical to string
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('Missing')

    # Define transformers
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    # Model pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_test_log = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred_test_log)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test_log))

    # Feature importance
    try:
        cat_features_out = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features_out = list(numeric_features) + list(cat_features_out)
        importances = model_pipeline.named_steps['regressor'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': all_features_out,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        feature_importance_df = pd.DataFrame(columns=['Feature', 'Importance'])

    # Retrain on full data for production
    model_pipeline.fit(X, y)

    return model_pipeline, numeric_features, categorical_features, r2, mae, feature_importance_df
