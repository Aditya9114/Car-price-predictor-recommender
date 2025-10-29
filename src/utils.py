# File: src/utils.py

import streamlit as st
import pandas as pd
import numpy as np
import locale
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# --- INDIAN NUMBER FORMATTING ---
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'en_IN')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'Indian')
        except locale.Error:
            pass 

def format_indian_number(n):
    """Formats a number using the Indian locale, with 0 decimal places."""
    if not isinstance(n, (int, float)):
        return n
    try:
        return locale.format_string("%.0f", n, grouping=True)
    except Exception:
        return f"{n:,.0f}"

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    """Loads and performs initial cleaning on the dataset."""
    try:
        df = pd.read_csv(file_path)
        
        df.rename(columns={
            'selling_price': 'Price',
            'vehicle_age': 'Vehicle Age',
            'km_driven': 'Kilometer',
            'fuel_type': 'Fuel Type',
            'transmission_type': 'Transmission',
            'brand': 'Make',
            'model': 'Model',
            'seller_type': 'Seller Type',
            'engine': 'Engine',
            'max_power': 'Max Power',
            'seats': 'Seats',
            'mileage': 'Mileage'
        }, inplace=True)

        df['Year'] = CURRENT_YEAR - df['Vehicle Age']
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year'], inplace=True) 
        df['Year'] = df['Year'].astype(int)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(subset=['Price'], inplace=True) 
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Make sure it's in the data/ folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- FEATURE ENGINEERING ---
def map_brand_to_tier(make):
    """Maps car makes to predefined tiers (Adjusted for cardekho_dataset)."""
    make_lower = str(make).lower()
    
    budget = ['maruti', 'hyundai', 'tata', 'renault', 'datsun', 'nissan', 'chevrolet', 'fiat']
    mid_range = ['honda', 'volkswagen', 'skoda', 'toyota', 'kia', 'mg', 'jeep', 'ford', 'mahindra', 'isuzu', 'mitsubishi', 'ssangyong']
    premium = ['bmw', 'audi', 'mercedes-benz', 'volvo', 'jaguar', 'lexus', 'mini', 'land rover']
    luxury = ['porsche', 'maserati', 'rolls-royce', 'ferrari', 'lamborghini', 'bentley']
    
    if make_lower in budget:
        return 'Budget'
    elif make_lower in mid_range:
        return 'Mid-Range'
    elif make_lower in premium:
        return 'Premium'
    elif make_lower in luxury:
        return 'Luxury'
    else:
        return 'Other'