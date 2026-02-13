"""
🛒 BASKET OPTIMIZATION - WEB APPLICATION
User selects products, system shows the cheapest supermarket
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Basket Optimization",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .best-market {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #28a745;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_data():
    """Load price matrix"""
    try:
        price_matrix = pd.read_csv('data/processed/price_matrix.csv', index_col=0)
        return price_matrix
    except FileNotFoundError:
        st.error("❌ Price matrix not found! Please run the notebook first.")
        return None

def optimize_basket(product_list, price_matrix, only_complete=True, penalty_method='exclude'):
    """Find cheapest supermarket for basket
    
    Args:
        product_list: List of product names
        price_matrix: Price matrix dataframe
        only_complete: If True, only show supermarkets with ALL products
        penalty_method: 'exclude', 'average', or 'highest' for missing products
    """
    selected_products = price_matrix.loc[price_matrix.index.isin(product_list)]
    
    supermarket_details = {}
    
    # Calculate average price for penalty
    avg_product_price = price_matrix.loc[price_matrix.index.isin(product_list)].mean().mean()
    
    for supermarket in selected_products.columns:
        prices = selected_products[supermarket].dropna()
        missing_count = len(product_list) - len(prices)
        
        # Skip if only_complete and market doesn't have all products
        if only_complete and missing_count > 0:
            continue
        
        if len(prices) > 0:
            total = prices.sum()
            
            # Apply penalty for missing products
            if missing_count > 0 and penalty_method == 'average':
                total += missing_count * avg_product_price
            elif missing_count > 0 and penalty_method == 'highest':
                max_price = price_matrix.loc[price_matrix.index.isin(product_list)].max().max()
                total += missing_count * max_price
            
            supermarket_details[supermarket] = {
                'total': total,
                'available_products': len(prices),
                'missing_products': missing_count,
                'products': prices.to_dict(),
                'has_penalty': missing_count > 0 and penalty_method != 'exclude'
            }
    
    results_df = pd.DataFrame([
        {
            'supermarket': sm,
            'total_price': details['total'],
            'available_products': details['available_products'],
            'missing_products': details['missing_products']
        }
        for sm, details in supermarket_details.items()
    ])
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('total_price').reset_index(drop=True)
    
    return results_df, supermarket_details

# Main application
def main():
    # Title
    st.markdown('<p class="main-header">🛒 BASKET OPTIMIZATION</p>', unsafe_allow_html=True)
    st.markdown("### Find the Cheapest Supermarket!")
    
    # Load data
    price_matrix = load_data()
    
    if price_matrix is None:
        st.stop()
    
    # Sidebar - Information
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=100)
        st.title("ℹ️ How It Works?")
        st.info("""
        1️⃣ Select products from the list below
        2️⃣ Click "Analyze Basket" button
        3️⃣ See the cheapest supermarket!
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Total Products", f"{len(price_matrix):,}")
        st.metric("Supermarkets", len(price_matrix.columns))
        
        st.markdown("---")
        st.markdown("**Supermarkets:**")
        for market in price_matrix.columns:
            st.write(f"🏪 {market}")
    
    # Main content
    st.markdown("---")
    
    # Product selection
    st.markdown("### 🛍️ Product Selection")
    
    # Search box
    search_term = st.text_input("🔍 Search Products (type name)", placeholder="e.g: milk, bread, cheese...")
    
    # Filtered products
    if search_term:
        filtered_products = [p for p in price_matrix.index if search_term.lower() in p.lower()]
    else:
        filtered_products = price_matrix.index.tolist()
    
    st.info(f"📦 Products found: **{len(filtered_products)}**")
    
    # Multiselect for product selection
    selected_products = st.multiselect(
        "Select products to add to your basket:",
        options=filtered_products,
        default=[],
        help="You can select multiple products"
    )
    
    # Options for optimization
    st.markdown("---")
    st.markdown("### ⚙️ Optimization Settings")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        only_complete = st.checkbox(
            "✅ Only show supermarkets with ALL products",
            value=True,
            help="Exclude supermarkets that don't have all selected products"
        )
    
    with col_opt2:
        if not only_complete:
            penalty_method = st.selectbox(
                "Missing product handling:",
                options=['exclude', 'average', 'highest'],
                index=0,
                help="How to handle missing products: exclude (no penalty), average price, or highest price"
            )
        else:
            penalty_method = 'exclude'
    
    # Show selected products
    if selected_products:
        st.markdown("---")
        st.markdown("### 📋 Your Basket")
        
        for i, product in enumerate(selected_products, 1):
            st.write(f"{i}. {product}")
        
        st.markdown("---")
        
        # Analysis button
        if st.button("🚀 Analyze Basket", type="primary"):
            
            with st.spinner("Analyzing..."):
                # Perform optimization
                results, details = optimize_basket(selected_products, price_matrix, only_complete, penalty_method)
                
                if len(results) == 0:
                    if only_complete:
                        st.error("❌ No supermarket has ALL selected products! Try unchecking 'Only show supermarkets with ALL products' option.")
                    else:
                        st.error("❌ Price information not found for these products!")
                else:
                    # Show results
                    st.success("✅ Analysis completed!")
                    
                    # Cheapest supermarket
                    best_market = results.iloc[0]['supermarket']
                    best_price = results.iloc[0]['total_price']
                    best_available = results.iloc[0]['available_products']
                    best_missing = results.iloc[0]['missing_products']
                    
                    # Check if best market has penalty
                    has_penalty = details[best_market].get('has_penalty', False)
                    
                    # Display in large box
                    penalty_note = ""
                    if has_penalty:
                        penalty_note = f"<p style='font-size: 0.9rem; color: #856404;'>⚠️ Price includes penalty for {best_missing} missing product(s)</p>"
                    
                    st.markdown(f"""
                    <div class="best-market">
                        <h1>🏆 CHEAPEST SUPERMARKET</h1>
                        <h2 style="color: #28a745; font-size: 3rem;">{best_market}</h2>
                        <h3>Total Price: £{best_price:.2f}</h3>
                        <p style="font-size: 1.2rem;">Available Products: {best_available}/{len(selected_products)}</p>
                        {penalty_note}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate savings
                    if len(results) > 1:
                        worst_price = results.iloc[-1]['total_price']
                        savings = worst_price - best_price
                        savings_pct = (savings / worst_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("💰 Cheapest", f"£{best_price:.2f}")
                        with col2:
                            st.metric("💸 Most Expensive", f"£{worst_price:.2f}")
                        with col3:
                            st.metric("📉 Savings", f"£{savings:.2f}", f"-{savings_pct:.1f}%")
                    
                    st.markdown("---")
                    
                    # Comparison of all supermarkets
                    st.markdown("### 📊 All Supermarkets Comparison")
                    
                    # Table
                    st.dataframe(
                        results.style.background_gradient(subset=['total_price'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors = ['#28a745' if sm == best_market else '#6c757d' for sm in results['supermarket']]
                    bars = ax.bar(results['supermarket'], results['total_price'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                    
                    # Write values on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'£{height:.2f}',
                                ha='center', va='bottom', fontweight='bold', fontsize=12)
                    
                    ax.set_xlabel('Supermarket', fontweight='bold', fontsize=14)
                    ax.set_ylabel('Total Price (£)', fontweight='bold', fontsize=14)
                    ax.set_title('Basket Price Comparison', fontweight='bold', fontsize=16)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Cheapest line
                    ax.axhline(y=best_price, color='#28a745', linestyle='--', linewidth=2, alpha=0.7, label=f'Cheapest: {best_market}')
                    ax.legend(fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Product-wise details
                    with st.expander("🔍 Product-wise Prices"):
                        selected_df = price_matrix.loc[price_matrix.index.isin(selected_products)]
                        st.dataframe(
                            selected_df.style.background_gradient(cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                    
                    # Missing product warnings
                    missing_warnings = []
                    for _, row in results.iterrows():
                        if row['missing_products'] > 0:
                            missing_warnings.append(f"⚠️ **{row['supermarket']}**: {row['missing_products']} products not found")
                    
                    if missing_warnings:
                        st.markdown("---")
                        st.warning("### ⚠️ Missing Product Notifications")
                        for warning in missing_warnings:
                            st.write(warning)
    
    else:
        st.info("👆 Please select products from above")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
        <p>🛒 Basket Optimization | UK Supermarket Price Analysis</p>
        <p>Made with <b>Group 36</b> using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
