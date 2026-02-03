import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(
    page_title="Market Pulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Market Pulse Dashboard")
st.markdown("### Real-Time Financial Data from FRED")

# Sidebar for API key and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    fred_api_key = st.text_input(
        "FRED API Key",
        type="password",
        help="Enter your FRED API key. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    
    st.markdown("---")
    
    # Date range selection
    st.subheader("Date Range")
    days_back = st.slider("Days of historical data", 30, 365, 180)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### About FRED
    Federal Reserve Economic Data (FRED) provides access to:
    - Economic indicators
    - Interest rates
    - Exchange rates
    - Commodity prices
    - And much more!
    
    [Get your free API key](https://fred.stlouisfed.org/docs/api/api_key.html)
    """)

# Function to fetch data from FRED
def fetch_fred_data(series_id, api_key, days_back=180):
    """Fetch data from FRED API"""
    if not api_key:
        return None
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching {series_id}: {str(e)}")
        return None

# Function to create line chart
def create_line_chart(df, title, color='#1f77b4'):
    """Create a plotly line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Function to calculate metrics
def calculate_metrics(df):
    """Calculate key metrics from dataframe"""
    if df is None or len(df) == 0:
        return None, None, None
    
    current_value = df['value'].iloc[-1]
    previous_value = df['value'].iloc[-2] if len(df) > 1 else current_value
    change = current_value - previous_value
    change_pct = (change / previous_value * 100) if previous_value != 0 else 0
    
    return current_value, change, change_pct

# Main dashboard
if not fred_api_key:
    st.warning("⚠️ Please enter your FRED API key in the sidebar to view data.")
    st.info("""
    ### How to get started:
    1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Enter your API key in the sidebar
    3. Explore real-time economic and financial data!
    """)
else:
    # Define data series to fetch
    data_series = {
        'SP500': {
            'id': 'SP500',
            'name': 'S&P 500',
            'color': '#00ff88',
            'category': 'Equities'
        },
        'DEXUSEU': {
            'id': 'DEXUSEU',
            'name': 'USD/EUR Exchange Rate',
            'color': '#3b82f6',
            'category': 'Foreign Exchange'
        },
        'GOLDAMGBD228NLBM': {
            'id': 'GOLDAMGBD228NLBM',
            'name': 'Gold Price (USD/oz)',
            'color': '#fbbf24',
            'category': 'Commodities'
        },
        'DCOILWTICO': {
            'id': 'DCOILWTICO',
            'name': 'Crude Oil WTI (USD/bbl)',
            'color': '#8b5cf6',
            'category': 'Commodities'
        },
        'DGS10': {
            'id': 'DGS10',
            'name': 'US 10-Year Treasury Rate (%)',
            'color': '#ef4444',
            'category': 'Fixed Income'
        },
        'CPIAUCSL': {
            'id': 'CPIAUCSL',
            'name': 'Consumer Price Index',
            'color': '#ec4899',
            'category': 'Economic Indicators'
        }
    }
    
    # Fetch all data
    with st.spinner('Fetching data from FRED...'):
        all_data = {}
        for key, series in data_series.items():
            df = fetch_fred_data(series['id'], fred_api_key, days_back)
            all_data[key] = df
    
    # Display metrics at the top
    st.markdown("### 📈 Current Market Snapshot")
    cols = st.columns(len(data_series))
    
    for idx, (key, series) in enumerate(data_series.items()):
        df = all_data[key]
        current, change, change_pct = calculate_metrics(df)
        
        with cols[idx]:
            if current is not None:
                st.metric(
                    label=series['name'],
                    value=f"{current:.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
            else:
                st.metric(
                    label=series['name'],
                    value="N/A",
                    delta=None
                )
    
    st.markdown("---")
    
    # Display charts in tabs by category
    categories = list(set(series['category'] for series in data_series.values()))
    tabs = st.tabs(categories + ["All Data"])
    
    # Organize by category
    for tab_idx, category in enumerate(categories):
        with tabs[tab_idx]:
            st.subheader(f"{category}")
            
            # Filter series by category
            category_series = {k: v for k, v in data_series.items() if v['category'] == category}
            
            # Create columns for charts
            if len(category_series) == 1:
                for key, series in category_series.items():
                    df = all_data[key]
                    if df is not None and len(df) > 0:
                        fig = create_line_chart(df, series['name'], series['color'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data available for {series['name']}")
            else:
                chart_cols = st.columns(2)
                for idx, (key, series) in enumerate(category_series.items()):
                    df = all_data[key]
                    with chart_cols[idx % 2]:
                        if df is not None and len(df) > 0:
                            fig = create_line_chart(df, series['name'], series['color'])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data available for {series['name']}")
    
    # All data tab - comparison view
    with tabs[-1]:
        st.subheader("📊 All Data - Normalized View")
        st.markdown("*All series normalized to 100 at the start date for comparison*")
        
        # Create normalized comparison chart
        fig = go.Figure()
        
        for key, series in data_series.items():
            df = all_data[key]
            if df is not None and len(df) > 0:
                # Normalize to 100
                normalized = (df['value'] / df['value'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=normalized,
                    mode='lines',
                    name=series['name'],
                    line=dict(color=series['color'], width=2),
                    hovertemplate=f"<b>{series['name']}</b><br>" +
                                "Date: %{x|%Y-%m-%d}<br>" +
                                "Normalized Value: %{y:.2f}<extra></extra>"
                ))
        
        fig.update_layout(
            title="All Series - Normalized Comparison (Base 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Value (Base 100)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Raw Data")
        
        # Create a combined dataframe for display
        if st.checkbox("Show raw data table"):
            selected_series = st.selectbox(
                "Select series to view",
                options=list(data_series.keys()),
                format_func=lambda x: data_series[x]['name']
            )
            
            df_display = all_data[selected_series]
            if df_display is not None:
                st.dataframe(
                    df_display[['date', 'value']].sort_values('date', ascending=False),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df_display[['date', 'value']].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_series}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Data provided by <a href='https://fred.stlouisfed.org/' target='_blank'>Federal Reserve Economic Data (FRED)</a></p>
    <p>Built with Streamlit • Updated in real-time</p>
</div>
""", unsafe_allow_html=True)