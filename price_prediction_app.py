import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UK Supermarket Price Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🛒 UK Supermarket Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Future/Past Price Prediction with Linear Regression</div>', unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model_and_data():
    """Load model, feature names and dataset"""
    try:
        model = joblib.load('models/linear_regression_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Load original data
        df_cleaned = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['capture_date'])
        df_engineered = pd.read_csv('data/processed/engineered_data.csv')
        unique_products = pd.read_csv('data/processed/unique_products.csv')
        
        return model, feature_names, df_cleaned, df_engineered, unique_products
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()

model, feature_names, df_cleaned, df_engineered, unique_products = load_model_and_data()

# Sidebar - Usage Information
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"""
    **Model:** Linear Regression  
    **R² Score:** 99.86%  
    **MAE:** 0.0162  
    **RMSE:** 0.0375  
    
    **Dataset:**
    - Total Records: {len(df_cleaned):,}
    - Number of Products: {len(unique_products):,}
    - Date Range: {df_cleaned['capture_date'].min().strftime('%d/%m/%Y')} - {df_cleaned['capture_date'].max().strftime('%d/%m/%Y')}
    """)
    
    st.header("ℹ️ How to Use?")
    st.markdown("""
    1. Select **Supermarket**
    2. Choose **Category**  
    3. Select **Product**
    4. Enter **Date** (past or future)
    5. Click **Predict** button
    """)
    
    st.header("🎯 Features")
    st.markdown("""
    ✅ 5 Supermarkets (ASDA, Tesco, Sainsbury's, Morrisons, Aldi)  
    ✅ 11 Categories  
    ✅ 3,784 Unique Products  
    ✅ Real-time Prediction  
    ✅ Visualization
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Product Selection")
    
    # Supermarket selection
    supermarkets = sorted(df_cleaned['supermarket_name'].unique())
    selected_supermarket = st.selectbox(
        "🏪 Select Supermarket:",
        options=supermarkets,
        index=0
    )
    
    # Category selection
    categories = sorted(df_cleaned[df_cleaned['supermarket_name'] == selected_supermarket]['category_name'].unique())
    selected_category = st.selectbox(
        "📦 Select Category:",
        options=categories,
        index=0
    )
    
    # Product selection (filter by selected supermarket and category)
    filtered_products = unique_products[
        (unique_products['supermarket_name'] == selected_supermarket) & 
        (unique_products['category_name'] == selected_category)
    ]['product_name'].sort_values().unique()
    
    if len(filtered_products) == 0:
        st.warning(f"⚠️ No products found for {selected_supermarket} - {selected_category} combination.")
        st.stop()
    
    selected_product = st.selectbox(
        "🛍️ Select Product:",
        options=filtered_products,
        index=0
    )
    
    # Date selection
    st.markdown("---")
    st.subheader("📅 Date Information")
    
    min_date = df_cleaned['capture_date'].min().date()
    max_date = df_cleaned['capture_date'].max().date()
    
    prediction_date = st.date_input(
        "Select Prediction Date:",
        value=max_date + timedelta(days=7),
        min_value=datetime(2024, 1, 1).date(),
        max_value=datetime(2025, 12, 31).date(),
        help="You can select a past or future date"
    )
    
    # Additional information
    st.info(f"""
    📊 **Selected Product Information:**
    - **Product:** {selected_product}
    - **Supermarket:** {selected_supermarket}
    - **Category:** {selected_category}
    - **Prediction Date:** {prediction_date.strftime('%d/%m/%Y')}
    """)

with col2:
    st.subheader("📈 Price Prediction")
    
    # Prediction button
    if st.button("🎯 PREDICT", use_container_width=True):
        with st.spinner("Predicting..."):
            # Find historical data for selected product
            product_data = df_cleaned[
                (df_cleaned['product_name'] == selected_product) & 
                (df_cleaned['supermarket_name'] == selected_supermarket) &
                (df_cleaned['category_name'] == selected_category)
            ]
            
            if len(product_data) == 0:
                st.error("❌ No historical data found for this product!")
                st.stop()
            
            # Get latest data (for reference)
            latest_data = product_data.sort_values('capture_date').iloc[-1]
            
            # Feature engineering (required for prediction)
            # Extract features from date
            pred_date = pd.to_datetime(prediction_date)
            month = pred_date.month
            day = pred_date.day
            day_of_week = pred_date.dayofweek
            week = pred_date.isocalendar()[1]
            is_weekend = 1 if day_of_week >= 5 else 0
            is_month_start = 1 if day <= 7 else 0
            is_month_end = 1 if day >= 25 else 0
            
            # Season (0: Winter, 1: Spring, 2: Summer, 3: Autumn)
            if month in [12, 1, 2]:
                season_encoded = 0
            elif month in [3, 4, 5]:
                season_encoded = 1
            elif month in [6, 7, 8]:
                season_encoded = 2
            else:
                season_encoded = 3
            
            # Supermarket one-hot encoding
            supermarket_features = {f'supermarket_{sm}': 0 for sm in ['ASDA', 'Aldi', 'Morrisons', 'Sains', 'Tesco']}
            if selected_supermarket == "Sainsbury's":
                supermarket_features['supermarket_Sains'] = 1
            else:
                supermarket_features[f'supermarket_{selected_supermarket}'] = 1
            
            # Category one-hot encoding
            category_features = {f'category_{cat}': 0 for cat in df_cleaned['category_name'].unique()}
            category_features[f'category_{selected_category}'] = 1
            
            # Other features
            price_unit_gbp = latest_data['price_unit_gbp']
            
            # Unit encoding (kg=0, l=1, unit=2)
            unit_map = {'kg': 0, 'l': 1, 'unit': 2}
            unit_encoded = unit_map.get(latest_data['unit'], 2)
            
            # Price category encoding (Ucuz=2, Orta=0, Pahalı=1)
            price_cat_map = {'Ucuz': 2, 'Orta': 0, 'Pahalı': 1}
            price_category_encoded = price_cat_map.get(latest_data.get('price_category', 'Orta'), 0)
            
            is_own_brand = latest_data.get('is_own_brand', 0)
            
            # Engineered features (average values - product's own average)
            price_to_unit_ratio = product_data['price_gbp'].mean() / (product_data['price_unit_gbp'].mean() + 0.001)
            price_vs_category_avg = 0  # Normalized değer
            price_vs_supermarket_avg = 0  # Normalized değer
            
            # Premium/discount features
            is_premium_category = 1 if selected_category in ['health_products', 'baby_products', 'home'] else 0
            is_discount_supermarket = 1 if selected_supermarket in ['Aldi', 'ASDA'] else 0
            premium_category_x_premium_supermarket = is_premium_category * (1 - is_discount_supermarket)
            
            # Create feature vector
            feature_dict = {
                'price_unit_gbp': price_unit_gbp,
                **supermarket_features,
                **category_features,
                'unit_encoded': unit_encoded,
                'price_category_encoded': price_category_encoded,
                'is_own_brand': is_own_brand,
                'month': month,
                'day': day,
                'day_of_week': day_of_week,
                'week': week,
                'is_weekend': is_weekend,
                'price_to_unit_ratio': price_to_unit_ratio,
                'price_vs_category_avg': price_vs_category_avg,
                'price_vs_supermarket_avg': price_vs_supermarket_avg,
                'is_month_start': is_month_start,
                'is_month_end': is_month_end,
                'season_encoded': season_encoded,
                'is_premium_category': is_premium_category,
                'is_discount_supermarket': is_discount_supermarket,
                'premium_category_x_premium_supermarket': premium_category_x_premium_supermarket
            }
            
            # Convert to DataFrame and sort (according to model feature order)
            X_pred = pd.DataFrame([feature_dict])
            X_pred = X_pred[feature_names]  # Same order as model
            
            # Make prediction
            predicted_price = model.predict(X_pred)[0]
            
            # Inverse scaling (if needed - previously scaled)
            # Price must be positive and in reasonable range
            actual_avg_price = product_data['price_gbp'].mean()
            actual_std_price = product_data['price_gbp'].std()
            
            # Convert predicted value to actual price scale
            final_predicted_price = predicted_price * actual_std_price + actual_avg_price
            final_predicted_price = max(0.01, final_predicted_price)  # Cannot be negative
            
            # Show results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 💰 Predicted Price")
            st.markdown(f"<h1 style='color: #1f77b4; font-size: 3rem;'>£{final_predicted_price:.2f}</h1>", unsafe_allow_html=True)
            st.markdown(f"**Date:** {prediction_date.strftime('%d %B %Y')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistics
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("📊 Average Price", f"£{actual_avg_price:.2f}", help="Historical average price of the product")
            
            with col_b:
                min_price = product_data['price_gbp'].min()
                max_price = product_data['price_gbp'].max()
                st.metric("📉 Lowest Price", f"£{min_price:.2f}")
            
            with col_c:
                st.metric("📈 Highest Price", f"£{max_price:.2f}")
            
            # Chart: Historical price trend + prediction
            st.markdown("---")
            st.subheader("📊 Price Trend and Prediction")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical prices
            product_data_sorted = product_data.sort_values('capture_date')
            ax.plot(product_data_sorted['capture_date'], 
                   product_data_sorted['price_gbp'], 
                   marker='o', linewidth=2, markersize=6, 
                   label='Historical Prices', color='steelblue')
            
            # Prediction point
            ax.scatter([pred_date], [final_predicted_price], 
                      s=300, color='red', marker='*', 
                      label=f'Prediction ({prediction_date.strftime("%d/%m/%Y")})', 
                      zorder=5, edgecolors='black', linewidth=1.5)
            
            # Labels
            ax.set_xlabel('Date', fontweight='bold', fontsize=12)
            ax.set_ylabel('Price (£)', fontweight='bold', fontsize=12)
            ax.set_title(f'{selected_product} - Price Trend', fontweight='bold', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed information (expander)
            with st.expander("🔍 Detailed Prediction Information"):
                st.write("**Features Used:**")
                st.json({
                    "Supermarket": selected_supermarket,
                    "Category": selected_category,
                    "Unit Price": f"£{price_unit_gbp:.2f}",
                    "Unit": latest_data['unit'],
                    "Own Brand": "Yes" if is_own_brand else "No",
                    "Month": month,
                    "Day": day,
                    "Day of Week": day_of_week,
                    "Week": week,
                    "Weekend": "Yes" if is_weekend else "No",
                    "Season": ["Winter", "Spring", "Summer", "Autumn"][season_encoded],
                    "Premium Category": "Yes" if is_premium_category else "No",
                    "Discount Supermarket": "Yes" if is_discount_supermarket else "No"
                })
                
                st.write("**Historical Price Statistics:**")
                st.dataframe(product_data_sorted[['capture_date', 'price_gbp', 'price_unit_gbp']].tail(10))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>📊 UK Supermarket Price Prediction System | Linear Regression Model (R²=99.86%)</p>
    <p>Data Source: UK Supermarket Prices Dataset (January-April 2024)</p>
</div>
""", unsafe_allow_html=True)
