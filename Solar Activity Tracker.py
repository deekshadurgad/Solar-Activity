"""
Solar Activity and Sunspot Cycle Study
A comprehensive Streamlit application for analyzing solar activity and sunspot cycles
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import StringIO
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Solar Activity & Sunspot Cycle Study",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #004E89;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    </style>
""", unsafe_allow_html=True)

# Data fetching functions
@st.cache_data(ttl=3600)
def fetch_sunspot_data():
    """Fetch sunspot data from SILSO (Sunspot Index and Long-term Solar Observations)"""
    try:
        # Monthly mean total sunspot number
        url = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data, sep=';', header=None, 
                           names=['Year', 'Month', 'Date', 'Sunspot_Number', 'Std_Dev', 'Observations', 'Marker'])
            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
            df = df[['Date', 'Sunspot_Number', 'Std_Dev']]
            return df
        else:
            return generate_sample_data()
    except Exception as e:
        st.warning(f"Could not fetch live data. Using sample data. Error: {str(e)}")
        return generate_sample_data()

@st.cache_data
def generate_sample_data():
    """Generate realistic sample sunspot data"""
    np.random.seed(42)
    dates = pd.date_range(start='1950-01-01', end='2024-12-01', freq='MS')
    
    # Simulate solar cycle with ~11-year period
    t = np.arange(len(dates))
    cycle_period = 132  # months (11 years)
    
    # Base cycle
    base_cycle = 80 * (np.sin(2 * np.pi * t / cycle_period) + 1)
    
    # Add harmonics
    harmonic = 20 * np.sin(4 * np.pi * t / cycle_period)
    
    # Add random noise
    noise = np.random.normal(0, 15, len(dates))
    
    # Combine
    sunspot_numbers = np.maximum(0, base_cycle + harmonic + noise)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sunspot_Number': sunspot_numbers,
        'Std_Dev': np.random.uniform(5, 15, len(dates))
    })
    
    return df

def calculate_solar_cycle_stats(df):
    """Calculate solar cycle statistics"""
    # Find peaks (maxima)
    peaks, properties = signal.find_peaks(df['Sunspot_Number'].values, 
                                         distance=80,  # ~7 years minimum
                                         prominence=50)
    
    # Find troughs (minima)
    troughs, _ = signal.find_peaks(-df['Sunspot_Number'].values, 
                                   distance=80,
                                   prominence=20)
    
    # Calculate cycle lengths
    if len(peaks) > 1:
        cycle_lengths = np.diff([df.iloc[p]['Date'] for p in peaks])
        avg_cycle_length = np.mean([cl.days / 365.25 for cl in cycle_lengths])
    else:
        avg_cycle_length = 11.0
    
    # Current cycle information
    current_value = df.iloc[-1]['Sunspot_Number']
    current_date = df.iloc[-1]['Date']
    
    # Determine phase
    recent_data = df.tail(36)  # Last 3 years
    trend = np.polyfit(range(len(recent_data)), recent_data['Sunspot_Number'].values, 1)[0]
    
    if trend > 0.5:
        phase = "Ascending (Solar Maximum Approaching)"
    elif trend < -0.5:
        phase = "Descending (Solar Minimum Approaching)"
    else:
        phase = "Near Extremum"
    
    return {
        'peaks': peaks,
        'troughs': troughs,
        'avg_cycle_length': avg_cycle_length,
        'current_value': current_value,
        'current_date': current_date,
        'phase': phase,
        'trend': trend
    }

def sine_function(x, amplitude, period, phase, offset):
    """Sine function for curve fitting"""
    return amplitude * np.sin(2 * np.pi * (x - phase) / period) + offset

def predict_future_activity(df, months_ahead=60):
    """Predict future solar activity using curve fitting"""
    # Use last 15 years of data for fitting
    recent_df = df.tail(180)
    x = np.arange(len(recent_df))
    y = recent_df['Sunspot_Number'].values
    
    try:
        # Fit sine curve
        initial_guess = [80, 132, 0, 80]
        params, _ = curve_fit(sine_function, x, y, p0=initial_guess, maxfev=5000)
        
        # Generate predictions
        future_x = np.arange(len(recent_df), len(recent_df) + months_ahead)
        predictions = sine_function(future_x, *params)
        predictions = np.maximum(0, predictions)  # No negative sunspot numbers
        
        # Create future dates
        last_date = df.iloc[-1]['Date']
        future_dates = pd.date_range(start=last_date + timedelta(days=30), 
                                     periods=months_ahead, freq='MS')
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Sunspot_Number': predictions
        })
        
        return future_df, params
    except:
        return None, None

def create_main_plot(df, stats, future_df=None):
    """Create the main time series plot"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Sunspot_Number'],
        mode='lines',
        name='Observed Sunspot Number',
        line=dict(color='#FF6B35', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Sunspot Number:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Add uncertainty band
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Sunspot_Number'] + df['Std_Dev'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Sunspot_Number'] - df['Std_Dev'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 107, 53, 0.2)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Mark peaks
    peak_dates = [df.iloc[p]['Date'] for p in stats['peaks']]
    peak_values = [df.iloc[p]['Sunspot_Number'] for p in stats['peaks']]
    
    fig.add_trace(go.Scatter(
        x=peak_dates,
        y=peak_values,
        mode='markers',
        name='Solar Maximum',
        marker=dict(color='red', size=10, symbol='star'),
        hovertemplate='<b>Solar Maximum</b><br>Date: %{x|%Y-%m}<br>Sunspot Number: %{y:.1f}<extra></extra>'
    ))
    
    # Mark troughs
    trough_dates = [df.iloc[t]['Date'] for t in stats['troughs']]
    trough_values = [df.iloc[t]['Sunspot_Number'] for t in stats['troughs']]
    
    fig.add_trace(go.Scatter(
        x=trough_dates,
        y=trough_values,
        mode='markers',
        name='Solar Minimum',
        marker=dict(color='blue', size=10, symbol='diamond'),
        hovertemplate='<b>Solar Minimum</b><br>Date: %{x|%Y-%m}<br>Sunspot Number: %{y:.1f}<extra></extra>'
    ))
    
    # Future predictions
    if future_df is not None:
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Predicted_Sunspot_Number'],
            mode='lines',
            name='Predicted',
            line=dict(color='green', width=2, dash='dash'),
            hovertemplate='<b>Predicted Date:</b> %{x|%Y-%m}<br><b>Predicted Sunspot Number:</b> %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Solar Activity: Sunspot Number Time Series',
        xaxis_title='Year',
        yaxis_title='Sunspot Number',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_cycle_distribution(df, stats):
    """Create histogram of sunspot numbers"""
    fig = px.histogram(
        df, 
        x='Sunspot_Number', 
        nbins=50,
        title='Distribution of Sunspot Numbers',
        labels={'Sunspot_Number': 'Sunspot Number', 'count': 'Frequency'},
        color_discrete_sequence=['#004E89']
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_yearly_average(df):
    """Create yearly average plot"""
    yearly_df = df.copy()
    yearly_df['Year'] = yearly_df['Date'].dt.year
    yearly_avg = yearly_df.groupby('Year')['Sunspot_Number'].mean().reset_index()
    
    fig = px.bar(
        yearly_avg,
        x='Year',
        y='Sunspot_Number',
        title='Yearly Average Sunspot Numbers',
        labels={'Sunspot_Number': 'Average Sunspot Number'},
        color='Sunspot_Number',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_fft_analysis(df):
    """Perform FFT analysis to identify periodicities"""
    # Resample to ensure uniform spacing
    sunspot_values = df['Sunspot_Number'].values
    
    # Perform FFT
    fft = np.fft.fft(sunspot_values)
    frequencies = np.fft.fftfreq(len(sunspot_values), d=1)  # Monthly data
    
    # Get positive frequencies only
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power = np.abs(fft[positive_freq_idx]) ** 2
    
    # Convert frequency to period (in years)
    periods = (1 / frequencies) / 12  # Convert months to years
    
    # Filter to reasonable solar cycle periods (5-20 years)
    valid_idx = (periods >= 5) & (periods <= 20)
    periods = periods[valid_idx]
    power = power[valid_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods,
        y=power,
        mode='lines',
        name='Power Spectrum',
        line=dict(color='#FF6B35', width=2)
    ))
    
    # Mark dominant period
    if len(power) > 0:
        dominant_period = periods[np.argmax(power)]
        fig.add_vline(x=dominant_period, line_dash="dash", line_color="red",
                     annotation_text=f"Dominant Period: {dominant_period:.1f} years")
    
    fig.update_layout(
        title='Frequency Analysis: Solar Cycle Periodicity',
        xaxis_title='Period (years)',
        yaxis_title='Power',
        template='plotly_white',
        height=400
    )
    
    return fig

# Main application
def main():
    st.markdown('<div class="main-header">Solar Activity & Sunspot Cycle Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyzing Solar Cycles and Predicting Future Activity</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        data_source = st.radio(
            "Data Source",
            ["Live Data (SILSO)", "Sample Data"],
            help="Choose between live data from SILSO or generated sample data"
        )
        
        st.markdown("---")
        
        show_predictions = st.checkbox("Show Predictions", value=True)
        
        if show_predictions:
            prediction_months = st.slider(
                "Prediction Window (months)",
                min_value=12,
                max_value=120,
                value=60,
                step=12
            )
        
        st.markdown("---")
        
        st.header("Display Options")
        show_distribution = st.checkbox("Show Distribution", value=True)
        show_yearly = st.checkbox("Show Yearly Average", value=True)
        show_fft = st.checkbox("Show Frequency Analysis", value=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### About
        This application analyzes solar activity through sunspot observations.
        
        **Data Source:** SILSO (Royal Observatory of Belgium)
        
        **Solar Cycle:** ~11 years average period
        """)
    
    # Load data
    with st.spinner("Loading solar activity data..."):
        if data_source == "Live Data (SILSO)":
            df = fetch_sunspot_data()
        else:
            df = generate_sample_data()
    
    # Calculate statistics
    stats = calculate_solar_cycle_stats(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Sunspot Number",
            value=f"{stats['current_value']:.1f}",
            delta=f"{stats['trend']:.2f} trend"
        )
    
    with col2:
        st.metric(
            label="Average Cycle Length",
            value=f"{stats['avg_cycle_length']:.1f} years"
        )
    
    with col3:
        st.metric(
            label="Solar Cycles Detected",
            value=f"{len(stats['peaks'])}"
        )
    
    with col4:
        st.metric(
            label="Current Phase",
            value=stats['phase'].split('(')[0].strip()
        )
    
    st.markdown("---")
    
    # Generate predictions if enabled
    future_df = None
    if show_predictions:
        with st.spinner("Generating predictions..."):
            future_df, params = predict_future_activity(df, prediction_months)
    
    # Main plot
    st.plotly_chart(
        create_main_plot(df, stats, future_df),
        use_container_width=True
    )
    
    # Additional analyses
    if show_distribution or show_yearly or show_fft:
        st.markdown("---")
        st.header("Detailed Analysis")
        
        tabs = []
        if show_distribution:
            tabs.append("Distribution")
        if show_yearly:
            tabs.append("Yearly Average")
        if show_fft:
            tabs.append("Frequency Analysis")
        
        tab_objects = st.tabs(tabs)
        
        tab_idx = 0
        if show_distribution:
            with tab_objects[tab_idx]:
                st.plotly_chart(create_cycle_distribution(df, stats), use_container_width=True)
            tab_idx += 1
        
        if show_yearly:
            with tab_objects[tab_idx]:
                st.plotly_chart(create_yearly_average(df), use_container_width=True)
            tab_idx += 1
        
        if show_fft:
            with tab_objects[tab_idx]:
                st.plotly_chart(create_fft_analysis(df), use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.header("Recent Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_rows = st.slider("Number of rows to display", 10, 100, 20)
    
    with col2:
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name=f"sunspot_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.dataframe(
        df.tail(display_rows).sort_values('Date', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Data provided by SILSO - Sunspot Index and Long-term Solar Observations</p>
        <p>Royal Observatory of Belgium, Brussels</p>
        <p>Created with Streamlit | Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
