# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity # Still needed for the button logic

# --- Import our functions from the src/ folder ---
from src.utils import load_data, format_indian_number, map_brand_to_tier, CURRENT_YEAR
from src.models import setup_recommendation_model, setup_price_prediction_model

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("🚗 Car Recommendation & Price Prediction App")

# --- 1. LOAD DATA ---
# Note the new file path
df_main = load_data('data/cardekho_dataset.csv')

if df_main is not None:
    # --- 2. SETUP OPTIONS & MODELS ---
    make_options = sorted(df_main['Make'].dropna().unique())
    fuel_type_options = sorted(df_main['Fuel Type'].dropna().unique())
    transmission_options = sorted(df_main['Transmission'].dropna().unique())
    seller_type_options = sorted(df_main['Seller Type'].dropna().unique())
    year_options = sorted(df_main['Year'].dropna().unique().astype(str), reverse=True)

    rec_cols = ['Make', 'Model', 'Year', 'Fuel Type', 'Transmission', 'Engine', 'Max Power']
    vectorizer_rec, feature_vectors_rec = setup_recommendation_model(df_main[rec_cols].copy())
    
    (price_pipeline, 
     numeric_features_price, 
     categorical_features_price, 
     r2_score_val, 
     mae_val, 
     feature_importance_df) = setup_price_prediction_model(df_main.copy())

    # --- 3. BUILD THE UI ---
    app_mode = option_menu(
        menu_title=None, 
        options=["Welcome", "Data Explorer", "Car Recommendations", "Predict Car Price"], 
        icons=["house-fill", "bar-chart-line-fill", "search", "cash-coin"], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )

    # --- Welcome Page ---
    if app_mode == "Welcome":
        st.header("Welcome to the Car App! 🚗")
        st.write("This application combines two machine learning models to help you:")
        st.subheader("1. Car Recommendations")
        st.write("Uses a **TF-IDF Vectorizer** and **Cosine Similarity**.")
        st.write("You describe your ideal car (brand, year, fuel, etc.), and it finds the 10 most *textually similar* cars from our dataset.")
        st.subheader("2. Predict Car Price")
        st.write("Uses a **Random Forest Regressor**.")
        st.write("You provide the exact specifications of a car, and it predicts a fair selling price based on the patterns learned from thousands of other car sales.")
        st.info(f"The app is trained on a dataset of **{format_indian_number(len(df_main))}** used cars.")

    # --- Data Explorer Page ---
    elif app_mode == "Data Explorer":
        st.header("Explore the Car Dataset")
        st.write(f"Displaying the first 100 rows of {format_indian_number(len(df_main))} total entries.")
        
        display_cols = ['Make', 'Model', 'Year', 'Price', 'Kilometer', 'Fuel Type', 
                        'Transmission', 'Seller Type', 'Engine', 'Max Power', 'Mileage', 'Seats']
        
        st.dataframe(
            df_main.head(100)[display_cols].style.format({
                "Price": lambda x: f"₹{format_indian_number(x)}",
                "Kilometer": lambda x: f"{format_indian_number(x)} km"
            })
        )

        st.subheader("Price Distribution")
        fig_price = px.histogram(df_main, x="Price", title="Distribution of Car Prices", nbins=50)
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Cars by Brand (Top 20)")
        brand_counts = df_main['Make'].value_counts().nlargest(20).reset_index()
        brand_counts.columns = ['Make', 'Count']
        fig_brand = px.bar(brand_counts, x='Make', y='Count', title="Top 20 Car Brands by Listing Count")
        st.plotly_chart(fig_brand, use_container_width=True)

    # --- Recommendation Page ---
    elif app_mode == "Car Recommendations":
        st.header("Find Similar Cars")
        st.write("Enter your preferences below. Leave 'Any' to ignore a field.")

        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox("Brand (Optional):", options=["Any"] + make_options, index=0)
            year_rec = st.selectbox("Year (Optional):", options=["Any"] + year_options, index=0)
            fuel_type_rec = st.selectbox("Fuel Type (Optional):", options=["Any"] + fuel_type_options, index=0)
        with col2:
            transmission_type_rec = st.selectbox("Transmission (Optional):", options=["Any"] + transmission_options, index=0)
            cc_rec = st.text_input("Engine CC (e.g., 1497):")
            bhp_rec = st.text_input("BHP (e.g., 117.3):")

        if st.button("Get Recommendations"):
            brand_str = "" if brand == "Any" else brand
            year_str = "" if year_rec == "Any" else year_rec
            fuel_str = "" if fuel_type_rec == "Any" else fuel_type_rec
            trans_str = "" if transmission_type_rec == "Any" else transmission_type_rec

            user_input_string = f"{brand_str} {year_str} {fuel_str} {trans_str} {cc_rec} {bhp_rec}".strip()

            if not user_input_string:
                st.warning("Please enter at least one preference.")
            else:
                try:
                    user_vector = vectorizer_rec.transform([user_input_string])
                    similarity_scores = cosine_similarity(user_vector, feature_vectors_rec).flatten()
                    top_10_indices = similarity_scores.argsort()[-10:][::-1]

                    recommended_cars = df_main.iloc[top_10_indices].copy()
                    recommended_cars['Similarity Score'] = [f"{score:.2f}" for score in similarity_scores[top_10_indices]]
                    display_columns = ['Make', 'Model', 'Year', 'Price', 'Fuel Type', 'Transmission', 'Engine', 'Max Power', 'Similarity Score']

                    st.subheader("Top 10 Recommended Cars:")
                    st.dataframe(
                        recommended_cars[display_columns].style.format({
                            "Price": lambda x: f"₹{format_indian_number(x)}"
                        }), 
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"An error occurred during recommendation: {e}")

    # --- Price Prediction Page ---
    elif app_mode == "Predict Car Price":
        st.header("Predict Your Car's Price")
        st.info(f"Model Performance: R-squared: **{r2_score_val:.2f}** | Mean Absolute Error: **₹{format_indian_number(mae_val)}**")
        
        with st.expander("See What Features Drive the Price"):
            st.subheader("Top 10 Most Important Features")
            st.dataframe(feature_importance_df.head(10), hide_index=True)
            
            fig_imp = px.bar(feature_importance_df.head(10), 
                             x='Importance', 
                             y='Feature', 
                             orientation='h', 
                             title="Top 10 Feature Importances")
            fig_imp.update_layout(yaxis_title="Feature", xaxis_title="Importance", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

        st.write("Enter the car's specifications:")

        make_options_with_placeholder = ["Select Make"] + make_options
        fuel_type_options_with_placeholder = ["Select Fuel Type"] + fuel_type_options
        transmission_options_with_placeholder = ["Select Transmission"] + transmission_options
        seller_type_options_with_placeholder = ["Select Seller Type"] + seller_type_options
        
        col1, col2 = st.columns(2)
        with col1:
            make_pred = st.selectbox("Make:", options=make_options_with_placeholder, index=0)
            model_pred = None 
            if make_pred != "Select Make":
                model_options_filtered = sorted(df_main[df_main['Make'] == make_pred]['Model'].dropna().unique())
                model_options_with_placeholder = ["Select Model"] + model_options_filtered
                model_pred = st.selectbox("Model:", options=model_options_with_placeholder, index=0)
            else:
                st.selectbox("Model:", options=["Select a Make first"], disabled=True)
            
            year_pred = st.number_input("Year of Manufacture:", min_value=1980, max_value=CURRENT_YEAR, step=1, placeholder="e.g., 2015")
            kilometer_pred = st.number_input("Kilometers Driven:", min_value=0.0, step=1000.0, placeholder="e.g., 50,000")
            mileage_pred = st.number_input("Mileage (km/l or km/kg):", min_value=0.0, step=1.0, placeholder="e.g., 20.5")

        with col2:
            fuel_type_pred = st.selectbox("Fuel Type:", options=fuel_type_options_with_placeholder, index=0)
            transmission_pred = st.selectbox("Transmission:", options=transmission_options_with_placeholder, index=0)
            seller_type_pred = st.selectbox("Seller Type:", options=seller_type_options_with_placeholder, index=0)
            engine_pred = st.number_input("Engine CC (e.g., 1497):", min_value=500.0, step=100.0, placeholder="e.g., 1200")
            max_power_pred = st.number_input("Max Power (BHP, e.g., 117.3):", min_value=30.0, step=10.0, placeholder="e.g., 80")
            seats_pred = st.number_input("Seats:", min_value=2, max_value=10, step=1, placeholder="e.g., 5")

        if st.button("Predict Price"):
            if (make_pred == "Select Make" or 
                model_pred is None or 
                model_pred == "Select Model" or 
                fuel_type_pred == "Select Fuel Type" or 
                transmission_pred == "Select Transmission" or 
                seller_type_pred == "Select Seller Type"):
                st.error("Please fill out all the dropdown fields.")
            elif not all([year_pred, kilometer_pred, engine_pred, max_power_pred, mileage_pred, seats_pred]):
                 st.error("Please ensure all numeric fields are filled.")
            else:
                try:
                    car_age = max(CURRENT_YEAR - year_pred, 0)
                    brand_tier = map_brand_to_tier(make_pred) # Use imported fn

                    user_data = pd.DataFrame({
                        'Kilometer':[kilometer_pred],
                        'Car Age':[car_age],
                        'Engine':[engine_pred],
                        'Max Power':[max_power_pred],
                        'Mileage': [mileage_pred], 
                        'Seats': [seats_pred], 
                        'Make':[make_pred],
                        'Model':[model_pred],
                        'Fuel Type':[fuel_type_pred],
                        'Transmission':[transmission_pred],
                        'Seller Type': [seller_type_pred], 
                        'Brand_Tier': [brand_tier]
                    }, columns=numeric_features_price + categorical_features_price) 

                    for col in categorical_features_price:
                            user_data[col] = user_data[col].astype(str)
                    for col in numeric_features_price:
                            user_data[col] = pd.to_numeric(user_data[col], errors='coerce')
                    user_data.fillna(0, inplace=True) 

                    log_price = price_pipeline.predict(user_data)
                    actual_price = np.expm1(log_price)[0]

                    st.success(f"Predicted Price: **₹{format_indian_number(actual_price)}**")

                except ValueError:
                    st.error("Please ensure all numeric fields have valid numbers.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Data could not be loaded. Please check the file path 'data/cardekho_dataset.csv' and format.")