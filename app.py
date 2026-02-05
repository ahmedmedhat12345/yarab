import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import os
from groq import Groq

# Load assets
@st.cache_resource
def load_models():
    clf = joblib.load('models/category_classifier.pkl')
    reg = joblib.load('models/price_regressor.pkl')
    cls_feats = joblib.load('models/cls_features.pkl')
    reg_feats = joblib.load('models/reg_features.pkl')
    market_stats = joblib.load('models/market_stats.pkl')
    return clf, reg, cls_feats, reg_feats, market_stats

def get_explanation(features, price, category, stats):
    median = stats['median_price']
    prompt = f"""
    Act as a real estate analyst.
    Data:
    - Valuation: ${price:,.0f} ({category})
    - Specs: {features['sqft_living']}sqft, {features['bedrooms']}bed, Zip-Rank {features['zip_rank']:.2f}
    - Market Median: ${median:,.0f}
    
    Output strictly:
    1. Reason for value (1 sentence).
    2. Verdict (Fair/High/Low).
    """
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key: return "⚠️ Groq Key missing."
        
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Insight unavailable."

# Main App
st.set_page_config(page_title="RE AI", layout="wide")
try:
    clf, reg, cls_cols, reg_cols, stats = load_models()
    city_map = stats.get('city_map', {})
    
except:
    st.error("Models not found. Please train models first using the notebook.")
    st.stop()

st.title("Real Estate Valuation AI 🏡")

# Sidebar
st.sidebar.header("Details")

# Location Inputs (Encoded)
# Location Inputs (Encoded)
if not city_map: city_map = {'Unknown': 0}

city = st.sidebar.selectbox("City", sorted(city_map.keys()))

sqft_living = st.sidebar.number_input("SqFt Living", 300, 10000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3) 
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) 
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
yr_renov = st.sidebar.number_input("Year Renovated", 0, 2025, 0)
view = st.sidebar.slider("View (0-4)", 0, 4, 0)
cond = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
water = st.sidebar.selectbox("Waterfront", [0, 1])
sqft_above = st.sidebar.number_input("SqFt Above", 0, 10000, 1500)
sqft_bsmt = st.sidebar.number_input("SqFt Basement", 0, 5000, 0)
sqft_lot = st.sidebar.number_input("SqFt Lot", 0, 50000, 5000)
floors = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, 0.5)

if st.sidebar.button("Valuate"):
    # Encode Input
    city_val = city_map.get(city, 0)
    zip_val = zip_map.get(zipcode, 0) # Median Price of Zip
    
    # Derived Logic
    house_age = 2026 - yr_built
    was_renovated = 1 if yr_renov > 0 else 0
    
    features = {
        'sqft_living': sqft_living,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'waterfront': water,
        'view': view,
        'condition': cond,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_bsmt,
        'sqft_lot': sqft_lot,
        'yr_built': yr_built,
        'yr_renovated': yr_renov,
        # Encoded Features
        'city_enc': city_val,
        'zip_rank': zip_val,
        'house_age': house_age,
        'was_renovated': was_renovated,
        'sqft_zip_interaction': sqft_living * zip_val 
    }
    
    # DataFrame for Model
    input_df = pd.DataFrame([features]).reindex(columns=reg_cols, fill_value=0)
    cls_input = pd.DataFrame([features]).reindex(columns=cls_cols, fill_value=0)

    c1, c2 = st.columns(2)
    
    try:
        log_price = reg.predict(input_df)[0]
        price = np.expm1(log_price)
        cat = clf.predict(cls_input)[0]
        
        c1.subheader(f"Type: {cat}")
        c2.metric("Valuation", f"${price:,.0f}")
        
        st.divider()
        with st.spinner("AI analyzing..."):
            st.info(get_explanation(features, price, cat, stats))
            
    except Exception as e:
        st.error(f"Error: {e}")
