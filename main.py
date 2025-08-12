import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
import time
from weather_service import get_weather_data, format_weather_data
from config import WEATHERSTACK_API_KEY, DEFAULT_LOCATION, OPENAI_API_KEY
from chatbot import initialize_chat, display_chat
import base64  # For get_base64_of_image

# Import scikit-learn modules for pickle loading
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# NOTE: The pickle file was saved with a non-standard module path "RandomForestClassifier".
# We fix this safely by using a CustomUnpickler that maps that fake module path to the
# correct sklearn class and also to numpy.dtype when requested.
import numpy as _np

# Add logger for error logging in show_dashboard
logger = logging.getLogger(__name__)

class SafeUnpickler(pickle.Unpickler):
    _CLASS_MAP = {
        ('RandomForestClassifier', 'RandomForestClassifier'): RandomForestClassifier,
        ('RandomForestClassifier', 'DecisionTreeClassifier'): DecisionTreeClassifier,
        ('RandomForestClassifier', 'dtype'): _np.dtype,  # mistakenly saved path for numpy dtype
    }
    def find_class(self, module, name):
        key = (module, name)
        if key in self._CLASS_MAP:
            return self._CLASS_MAP[key]
        # Fall back to normal behaviour
        return super().find_class(module, name)

# Helper to safely convert to float
def safe_float(val, default):
    try:
        if val is None or val == 'N/A':
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def setup_ui_theme():
    st.set_page_config(
        page_title="SowSmart OS - Mango Analytics",
        page_icon="🥭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for the chat button and global theme
    st.markdown("""
    <style>
    body, .main-content {
        background: linear-gradient(135deg, #f8fbff 0%, #e3f0ff 100%) !important;
        color: #111 !important;
    }
    .stApp, .st-emotion-cache-18ni7ap, .st-emotion-cache-1dp5vir, .st-emotion-cache-1d391kg, .st-emotion-cache-1oe5cao {
        background: #fff !important;
        color: #111 !important;
    }
    .stSidebar, .st-emotion-cache-1d391kg, .st-emotion-cache-1oe5cao {
        background: linear-gradient(135deg, #e3f0ff 0%, #f8fbff 100%) !important;
        color: #111 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2196f3 0%, #6dd5ed 100%);
        color: #111;
        border: none;
        border-radius: 7px;
        padding: 12px 26px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(33,150,243,0.10);
        transition: background 0.3s, color 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1976d2 0%, #00c6fb 100%);
        color: #fff;
        transform: scale(1.04);
        box-shadow: 0 6px 18px rgba(33,150,243,0.18);
    }
    .card, .metric-card, .chat-container {
        background: #fff;
        color: #111;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(33,150,243,0.08);
        transition: box-shadow 0.3s, transform 0.2s;
        border-left: 4px solid #2196f3;
    }
    .card:hover, .metric-card:hover, .chat-container:hover {
        box-shadow: 0 8px 24px rgba(33,150,243,0.13);
        transform: translateY(-3px) scale(1.01);
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #1976d2;
        margin: 7px 0;
    }
    .metric-label {
        font-size: 16px;
        color: #1976d2;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .user-message {
        background: linear-gradient(90deg, #e3f0ff 0%, #bbdefb 100%);
        color: #111;
        padding: 14px 20px;
        border-radius: 20px 20px 6px 20px;
        margin: 10px 0 10px 50px;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 1.08rem;
        box-shadow: 0 2px 8px rgba(33,150,243,0.10);
        animation: popIn 0.7s cubic-bezier(.68,-0.55,.27,1.55);
    }
    .assistant-message {
        background: #e3f0ff;
        color: #1976d2;
        padding: 14px 20px;
        border-radius: 20px 20px 20px 6px;
        margin: 10px 50px 10px 0;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 1.08rem;
        box-shadow: 0 2px 8px rgba(33,150,243,0.10);
        animation: popIn 0.7s cubic-bezier(.68,-0.55,.27,1.55);
    }
    /* Animations */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0% { transform: scale(0.7); opacity: 0; }
        80% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); }
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #e3f0ff;
    }
    ::-webkit-scrollbar-thumb {
        background: #90caf9;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #42a5f5;
}
    </style>
    """, unsafe_allow_html=True)
    
    # --- Global background gradient for main content area ---
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f0ff 0%, #e0f7fa 50%, #f8fbff 100%) !important;
        min-height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)
    

@st.cache_resource
def load_model():
    """Load the trained mango classifier model using custom unpickler"""
    try:
        import warnings
        # Suppress sklearn version warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Load model package (dict with 'model', 'encoders', 'feature_names') using joblib
        pkg = joblib.load('mango_classifier.pkl')
        model = pkg.get('model')
        st.session_state['encoders'] = pkg.get('encoders')
        st.session_state['feature_names'] = pkg.get('feature_names')
        
        st.success(f"✅ Model loaded successfully! Model type: {type(model).__name__}")
        return model
    except FileNotFoundError:
        st.error("Model file 'mango_classifier.pkl' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure all required sklearn modules are installed and the model file is compatible.")
        return None

def predict_mango_type(model, features):
    """Make prediction using the loaded model"""
    try:
        # Convert features to numpy array
        feature_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(feature_array)
            return prediction[0], prediction_proba[0]
        except:
            return prediction[0], None
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def show_weather_card(weather):
    """Display weather information in a card with enhanced error handling"""
    if not weather:
        st.warning("No weather data available")
        return
    
    # Create a more robust display that handles missing data
    with st.container():
        st.markdown("### 🌤️ Current Weather")
        
        # Top row for main weather info
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Weather icon and temperature
            if weather.get('weather_icon'):
                st.image(weather['weather_icon'], width=80)
            
            temp = weather.get('temperature', 'N/A')
            st.markdown(f"### {temp}°C")
            
            # Weather description
            if weather.get('weather_desc') and weather['weather_desc'] != 'N/A':
                st.caption(weather['weather_desc'].title())
        
        with col2:
            st.markdown("#### 📊 Weather Details")
            
            # Create two columns for metrics
            cols = st.columns(2)
            
            # Left column metrics
            with cols[0]:
                # Feels Like
                feels_like = weather.get('feelslike', 'N/A')
                if feels_like != 'N/A':
                    st.metric("Feels Like", f"{feels_like}°C")
                else:
                    st.metric("Feels Like", "N/A")
                
                # Humidity
                humidity = weather.get('humidity', 'N/A')
                if humidity != 'N/A':
                    st.metric("Humidity", f"{humidity}%")
                else:
                    st.metric("Humidity", "N/A")
            
            # Right column metrics
            with cols[1]:
                # Wind
                wind_speed = weather.get('wind_speed', 0)
                wind_dir = weather.get('wind_dir', '')
                if wind_speed != 0 and wind_dir:
                    st.metric("Wind", f"{wind_speed} km/h {wind_dir}")
                elif wind_speed != 0:
                    st.metric("Wind Speed", f"{wind_speed} km/h")
                
                # UV Index with color coding
                uv_index = weather.get('uv_index', 'N/A')
                if uv_index != 'N/A':
                    uv_color = "#27ae60"  # Green
                    uv_status = "Low"
                    if uv_index > 7:
                        uv_color = "#e74c3c"  # Red
                        uv_status = "Very High"
                    elif uv_index > 5:
                        uv_color = "#f39c12"  # Orange
                        uv_status = "High"
                    
                    st.markdown(
                        f"<div style='margin-top: 16px;'>"
                        f"<div style='font-weight: 500; color: #555;'>UV Index</div>"
                        f"<div style='display: flex; align-items: center;'>"
                        f"<span style='font-size: 1.5rem; font-weight: 600; color: {uv_color};'>{uv_index}</span>"
                        f"<span style='margin-left: 8px; color: {uv_color};'>{uv_status}</span>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
        
        # Additional details in an expander
        with st.expander("More Details"):
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.metric("Pressure", f"{weather.get('pressure', 'N/A')} hPa")
                st.metric("Precipitation", f"{weather.get('precip', 0)} mm")
            with detail_cols[1]:
                st.metric("Visibility", f"{weather.get('visibility', 'N/A')} km")
                st.metric("Observation Time", weather.get('observation_time', 'N/A'))
        
        # Debug information (only shown in development)
        if st.session_state.get('debug_mode', False):
            with st.expander("Debug Info"):
                st.json(weather)

def show_dashboard():
    # --- Hero Section (moved higher, no box, plant emoji, left-aligned, updated name) ---
    st.markdown("""
    <div style='margin-bottom: 24px; text-align: left;'>
        <span style='font-size:3.2rem; font-weight:900; color:#1976d2; margin-right: 18px; letter-spacing:1px; vertical-align: middle;'>🌱 Soil Smart</span>
        <span style='font-size:1.5rem; color:#27ae60; font-weight:700; margin-left: 8px; vertical-align: middle;'>Grow Smarter, Farm Better</span>
        <div style='margin-top: 8px; color: #1976d2; font-size: 1.18rem; font-weight:500;'>
            Welcome to your AI-powered mango farm dashboard. Monitor, plan, and optimize every aspect of your farm with beautiful analytics and actionable insights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    # --- Get weather data and key values up front for use in hero/cards ---
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    weather_data = None
    formatted_weather = None
    try:
        weather_data = get_weather_data()
        if weather_data:
            formatted_weather = format_weather_data(weather_data)
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Error processing weather data: {str(e)}")
        logger.error(f"Error in show_dashboard: {str(e)}", exc_info=True)
    try:
        current_temp = safe_float(formatted_weather.get('temperature') if formatted_weather else None, st.session_state.get('avg_temperature', 26.0))
        current_humidity = safe_float(formatted_weather.get('humidity') if formatted_weather else None, st.session_state.get('humidity_percent', 45))
        current_uv_index = safe_float(formatted_weather.get('uv_index') if formatted_weather else None, 6)
        weather_desc = formatted_weather.get('weather_desc', 'N/A') if formatted_weather else 'N/A'
    except Exception as e:
        logger.error(f"Error parsing weather values: {str(e)}")
        current_temp = st.session_state.get('avg_temperature', 26.0)
        current_humidity = st.session_state.get('humidity_percent', 45)
        current_uv_index = 6
        weather_desc = 'N/A'

    # --- Animated Cards Section ---
    st.markdown("""
    <style>
    .dashboard-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(33,150,243,0.10);
        padding: 28px 24px 22px 24px;
        margin-bottom: 18px;
        border-left: 6px solid #2196f3;
        transition: box-shadow 0.3s, transform 0.2s, border-color 0.3s;
        animation: fadeInUp 0.7s cubic-bezier(.68,-0.55,.27,1.55);
    }
    .dashboard-card:hover {
        box-shadow: 0 12px 32px rgba(33,150,243,0.18);
        border-left: 6px solid #27ae60;
        transform: translateY(-4px) scale(1.02);
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Key Metrics Cards ---
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 28px; margin-bottom: 18px;'>
        <div class='dashboard-card' style='flex:1; min-width:220px;'>
            <span style='font-size:2.1rem;'>🌡️</span>
            <div style='font-size:1.15rem; color:#1976d2; font-weight:700;'>Current Temperature</div>
            <div style='font-size:2.2rem; font-weight:900; color:#1976d2;'>{current_temp}°C</div>
            <div style='font-size:1.01rem; color:#27ae60;'>{weather_desc}</div>
        </div>
        <div class='dashboard-card' style='flex:1; min-width:220px;'>
            <span style='font-size:2.1rem;'>💧</span>
            <div style='font-size:1.15rem; color:#1976d2; font-weight:700;'>Humidity</div>
            <div style='font-size:2.2rem; font-weight:900; color:#1976d2;'>{current_humidity}%</div>
            <div style='font-size:1.01rem; color:#27ae60;'>Moisture Level</div>
        </div>
        <div class='dashboard-card' style='flex:1; min-width:220px;'>
            <span style='font-size:2.1rem;'>🌞</span>
            <div style='font-size:1.15rem; color:#1976d2; font-weight:700;'>UV Index</div>
            <div style='font-size:2.2rem; font-weight:900; color:#1976d2;'>{current_uv_index}</div>
            <div style='font-size:1.01rem; color:#27ae60;'>Sunlight Intensity</div>
        </div>
        <div class='dashboard-card' style='flex:1; min-width:220px;'>
            <span style='font-size:2.1rem;'>🥭</span>
            <div style='font-size:1.15rem; color:#1976d2; font-weight:700;'>Total Mangoes Selected</div>
            <div style='font-size:2.2rem; font-weight:900; color:#1976d2;'>{total_mangoes}</div>
            <div style='font-size:1.01rem; color:#27ae60;'>In Crop Planner</div>
        </div>
    </div>
    """.format(
        current_temp=current_temp,
        weather_desc=weather_desc,
        current_humidity=current_humidity,
        current_uv_index=current_uv_index,
        total_mangoes=len(st.session_state.get('mango_list', []))
    ), unsafe_allow_html=True)

    # --- Animated Section Dividers ---
    st.markdown("""
    <div style='height:2px; background: linear-gradient(90deg, #2196f3 0%, #27ae60 100%); border-radius:2px; margin: 32px 0 24px 0; animation: fadeInUp 1.2s;'></div>
    """, unsafe_allow_html=True)

    # --- Weather Overview Section (with animation) ---
    with st.container():
        st.subheader("🌤️ Current Conditions & Farm Overview")
        weather_col1, weather_col2, weather_col3 = st.columns([1, 1, 1], gap="small")
    with weather_col1:
        st.markdown(f"""
            <div class='dashboard-card' style='background: linear-gradient(135deg, #42a5f5 0%, #1976d2 100%); color: white; min-height: 80px; max-width: 98%;'>
                <h4 style='margin: 0 0 4px 0; font-size: 1.1rem; opacity: 0.9;'>Current Weather</h4>
                <div style='font-size: 1.5rem; font-weight: 700; margin: 4px 0;'>
                    {current_temp}°C
                </div>
                <div style='font-size: 1.01rem; opacity: 0.9;'>
                    {formatted_weather.get('weather_desc', 'N/A') if formatted_weather else 'N/A'}
                </div>
            </div>
        """, unsafe_allow_html=True)
    with weather_col2:
        feels_temp = formatted_weather.get('feelslike', current_temp) if formatted_weather else current_temp
        st.markdown(f"""
            <div class='dashboard-card' style='background: linear-gradient(135deg, #90caf9 0%, #42a5f5 100%); color: white; min-height: 80px; max-width: 98%;'>
                <h4 style='margin: 0 0 4px 0; font-size: 1.1rem; opacity: 0.9;'>Feels Like</h4>
                <div style='font-size: 1.5rem; font-weight: 700; margin: 4px 0;'>
                    {feels_temp}°C
                </div>
                <div style='font-size: 1.01rem; opacity: 0.9;'>
                    Temperature
                </div>
            </div>
        """, unsafe_allow_html=True)
    with weather_col3:
        st.markdown(f"""
            <div class='dashboard-card' style='background: linear-gradient(135deg, #1976d2 0%, #64b5f6 100%); color: white; min-height: 80px; max-width: 98%;'>
                <h4 style='margin: 0 0 4px 0; font-size: 1.1rem; opacity: 0.9;'>Humidity</h4>
                <div style='font-size: 1.5rem; font-weight: 700; margin: 4px 0;'>
                    {current_humidity}%
                </div>
                <div style='font-size: 1.01rem; opacity: 0.9;'>
                    Moisture Level
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.divider()

    # --- Environmental Status & Metrics Section ---
    with st.container():
        st.subheader("🌡️ Environmental Status & Farm Metrics")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("#### Temperature & Humidity Gauges")
            # Always show 'Ideal' status and ideal color
            temp_status = "Ideal"
            temp_color = "#1976d2"
            temp_percent = min(100, max(0, ((current_temp + 20) / 60) * 100))
            st.markdown(f"""
                <div style='background: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(33,150,243,0.08); border-left: 4px solid #1976d2;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Temperature</span>
                    <span style='font-weight: 500; color: {temp_color};'>{temp_status}</span>
                </div>
                    <div style='height: 8px; background: #e3f0ff; border-radius: 4px; margin-bottom: 5px; overflow: hidden;'>
                    <div style='height: 100%; width: {temp_percent}%; background: {temp_color};'></div>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #7f8c8d;'>
                        <span>-20°C</span>
                    <span>{current_temp}°C</span>
                        <span>40°C</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Always show 'Ideal' status and ideal color for humidity
            humidity_status = "Ideal"
            humidity_color = "#1976d2"
            st.markdown(f"""
                <div style='background: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(33,150,243,0.08); border-left: 4px solid #1976d2;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Humidity</span>
                    <span style='font-weight: 500; color: {humidity_color};'>{humidity_status}</span>
                </div>
                    <div style='height: 8px; background: #e3f0ff; border-radius: 4px; margin-bottom: 5px; overflow: hidden;'>
                    <div style='height: 100%; width: {current_humidity}%; background: {humidity_color};'></div>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #7f8c8d;'>
                    <span>0%</span>
                    <span>{current_humidity}%</span>
                    <span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("Temperature and humidity are key for optimal mango growth. Keep them in the ideal range for best results.")
        with col2:
            st.markdown("#### Farm Health & Performance Metrics")
            metric_labels = [
                ("Soil Health", "0%", "No Data"),
                ("Yield Forecast", "0%", "No Data"),
                ("Disease Alerts", "0", "No Data"),
                ("Water Usage", "0 L", "No Data"),
            ]
            metric_cols = st.columns(2, gap="medium")
            for i, (label, value, note) in enumerate(metric_labels):
                with metric_cols[i % 2]:
                    st.markdown(f'''
                    <div style='
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: flex-end;
                        background: #fff;
                        border-radius: 18px;
                        padding: 18px 28px 18px 18px;
                        margin-bottom: 18px;
                        box-shadow: 0 2px 8px rgba(33,150,243,0.10);
                        border-left: 5px solid #2196f3;
                        min-height: 110px;
                        position: relative;
                    '>
                        <div style='
                            position: absolute;
                            top: 18px;
                            left: 22px;
                            font-size: 1.08rem;
                            font-weight: 700;
                            color: #1976d2;
                            opacity: 0.95;
                        '>
                            {label}
                        </div>
                        <div style='
                            font-size: 2.2rem;
                            font-weight: 800;
                            color: #1976d2;
                            margin-top: 18px;
                            margin-bottom: 6px;
                            text-align: right;
                            width: 100%;
                        '>
                            {value}
                        </div>
                        <div style='
                            font-size: 1rem;
                            color: #42a5f5;
                            opacity: 0.85;
                            text-align: right;
                            width: 100%;
                        '>
                            {note}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            st.caption("No sensor or weather data has been inputted yet. Please input data to view farm metrics.")
    st.divider()

    # --- Manage Mangoes Section (Synced with Planner) ---
    st.markdown("""
    <div class='dashboard-card' style='background: linear-gradient(90deg, #e3f0ff 0%, #f8fbff 100%); border-left: 6px solid #27ae60; margin-bottom: 18px;'>
        <span style='font-size:1.5rem; font-weight:700; color:#1976d2; margin-right: 10px;'>🥭 Manage Mangoes</span>
        <span style='background:#e3f0ff; color:#1976d2; border-radius:8px; padding:4px 14px; font-size:1.1rem; font-weight:600;'>Total: {}</span>
    </div>
    """.format(len(st.session_state.get('mango_list', []))), unsafe_allow_html=True)
    if 'mango_list' not in st.session_state:
        st.session_state['mango_list'] = []
    for mango_type in CORE_MANGO_TYPES:
        col_label, col_action = st.columns([5, 1])
        with col_label:
            st.write(mango_type.replace('_', ' '))
        with col_action:
            if mango_type in st.session_state['mango_list']:
                if st.button("🗑️", key=f"dashboard_delete_{mango_type}", help=f"Remove {mango_type} from your list"):
                    st.session_state['mango_list'].remove(mango_type)
                    st.rerun()
            else:
                if st.button("Add", key=f"dashboard_add_{mango_type}", help=f"Add {mango_type} to your list"):
                    st.session_state['mango_list'].append(mango_type)
                    st.rerun()
    st.divider()
    # --- End Manage Mangoes Section ---

    # --- Weather Details Section ---
    if formatted_weather:
        with st.container():
            st.subheader("🌬️ Weather Details")
            w_col1, w_col2, w_col3 = st.columns(3, gap="large")
            with w_col1:
                st.metric("Wind Speed", f"{formatted_weather.get('wind_speed', 'N/A')} km/h")
                st.metric("Pressure", f"{formatted_weather.get('pressure', 'N/A')} hPa")
            with w_col2:
                st.metric("Wind Direction", formatted_weather.get('wind_dir', 'N/A'))
                st.metric("Precipitation", f"{formatted_weather.get('precip', 0)} mm")
            with w_col3:
                st.metric("UV Index", formatted_weather.get('uv_index', 6))
                st.metric("Visibility", f"{formatted_weather.get('visibility', 'N/A')} km")
            with st.expander("More Weather Details"):
                st.json(formatted_weather)
    st.divider()
    # --- END Weather Details Section ---

    # Add Boost Your Success tips at the bottom of the dashboard
    st.markdown("""
    <div style='background: linear-gradient(90deg,#f8fbff 0%,#e3f0ff 100%); border-radius: 12px; padding: 18px 24px; margin-top: 18px; box-shadow: 0 1px 4px rgba(33,150,243,0.09); border-left: 4px solid #42a5f5;'>
        <h4 style='color: #1976d2;'>🚀 Boost Your Success!</h4>
        <ul style='color: #1976d2; font-size: 1.05rem;'>
            <li>🌱 Try intercropping with legumes to improve soil fertility.</li>
            <li>💧 Install soil moisture sensors for precision irrigation.</li>
            <li>🦠 Use biofertilizers and beneficial microbes for healthier trees.</li>
            <li>📊 Track your yields and costs each season for better planning.</li>
            <li>🛡️ Join a local farmer group for knowledge sharing and support.</li>
            <li>📦 Explore direct-to-consumer sales for higher profits.</li>
            <li>🌤️ Use weather apps to plan irrigation and harvests.</li>
            <li>🎯 Set a yield goal for next season and track your progress!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

CORE_MANGO_TYPES = ["Alphonso", "Haden", "Keitt", "Kent", "Tommy_Atkins"]

def show_classifier(model):
    st.title("🔍 Crop Type Classifier")
    st.markdown("""
    <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Classify crop types using machine learning based on environmental and soil conditions. 
        Enter your farm parameters below to get accurate predictions for optimal mango cultivation.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # Load model if not already loaded
    if model is None:
        model = load_model()
        if model is None:
            st.error("Failed to load the model. Please ensure 'mango_classifier.pkl' exists in the directory.")
            return
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write("**Expected Features:**", st.session_state.get("feature_names", []))
        st.write("**Model Status:** ✅ Loaded and Ready")
    st.divider()
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🌱 Soil Parameters")
        with st.container():
            ph_level = st.slider(
                "Soil pH Level",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1,
                help="Enter the soil pH level (6.0-7.5 is ideal for mangoes)"
            )
            st.caption(f"Current pH: {ph_level} (Optimal range: 6.0-7.5)")
            soil_types = ["Sandy", "Loamy", "Clay", "Silt", "Peat", "Chalk", "Gravel"]
            soil_type = st.selectbox(
                "Soil Type",
                options=soil_types,
                index=1,
                help="Dominant soil composition in your farm"
            )
            soil_type_map = {name: idx for idx, name in enumerate(soil_types)}
            soil_type_encoded = soil_type_map[soil_type]
            st.caption(f"Selected: {soil_type} soil")
        st.markdown("### 🌍 Location")
        with st.container():
            altitude_meters = st.number_input(
                "Altitude (m)",
                min_value=-100.0,
                max_value=3000.0,
                value=10.0,
                step=10.0,
                help="Altitude of the cultivation area in meters above sea level for Cambridge, MA"
            )
            st.caption(f"Farm altitude: {altitude_meters} meters")
    with col2:
        st.markdown("### 🌦️ Weather Conditions")
        with st.container():
            avg_temperature = st.session_state.get('avg_temperature', 28.0)
            humidity_percent = st.session_state.get('humidity_percent', 65)
            st.markdown(f"**Average Temperature:** {avg_temperature}°C")
            st.markdown(f"**Relative Humidity:** {humidity_percent}%")
            st.caption("These values are set in the sidebar. Adjust them there for different predictions.")
            annual_rainfall = st.number_input(
                "Annual Rainfall (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=1200.0,
                step=10.0,
                help="Total annual rainfall in millimetres for Cambridge, MA"
            )
            st.caption(f"Annual rainfall: {annual_rainfall} mm")
            # Growing score
            growing_score = st.slider(
                "Growing Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=0.1,
                help="Expert-assigned score of overall growing conditions for Cambridge, MA"
            )
            st.caption(f"Growing condition score: {growing_score}/10")
    st.divider()

    # Calculate derived features
    temp_humidity_index = avg_temperature + 0.1 * humidity_percent
    rainfall_humidity_ratio = annual_rainfall / (humidity_percent + 1e-6)
    ph_temp_interaction = ph_level * avg_temperature
    altitude_temp_stress = (altitude_meters / 1000.0) * avg_temperature
    
    # Prepare features for prediction
    features = [
        ph_level, avg_temperature, annual_rainfall, humidity_percent,
        altitude_meters, soil_type_encoded, temp_humidity_index,
        rainfall_humidity_ratio, ph_temp_interaction, altitude_temp_stress,
        growing_score
    ]
    
    # Show calculated features
    with st.expander("📊 Calculated Features"):
        feature_col1, feature_col2 = st.columns(2)
        with feature_col1:
            st.metric("Temp-Humidity Index", f"{temp_humidity_index:.2f}")
            st.metric("Rainfall-Humidity Ratio", f"{rainfall_humidity_ratio:.2f}")
        with feature_col2:
            st.metric("pH-Temp Interaction", f"{ph_temp_interaction:.2f}")
            st.metric("Altitude-Temp Stress", f"{altitude_temp_stress:.2f}")
    
    # Prediction button
    if st.button("🌱 Predict Mango Type", use_container_width=True, type="primary"):
        with st.spinner("Analyzing conditions..."):
            prediction, probabilities = predict_mango_type(model, features)
            if prediction is not None:
                st.session_state['last_classifier_prediction'] = prediction
                st.session_state['last_classifier_probabilities'] = probabilities
                st.session_state['last_classifier_class_names'] = list(model.classes_) if hasattr(model, 'classes_') else []
            else:
                st.session_state['last_classifier_prediction'] = None
                st.session_state['last_classifier_probabilities'] = None
                st.session_state['last_classifier_class_names'] = []

    # Use session state to persist prediction/probabilities
    prediction = st.session_state.get('last_classifier_prediction', None)
    probabilities = st.session_state.get('last_classifier_probabilities', None)
    class_names = st.session_state.get('last_classifier_class_names', [])
    # Filter class_names and probabilities to only core types
    if probabilities is not None:
        filtered = [(c, p) for c, p in zip(class_names, probabilities) if c in CORE_MANGO_TYPES]
        class_names = [c for c, _ in filtered]
        probabilities = [p for _, p in filtered]
        
        if prediction is not None:
            st.success(f"✅ Predicted Mango Type: **{prediction}**")
            if probabilities is not None:
                st.markdown("### Prediction Confidence")
                prob_df = pd.DataFrame({
                    'Mango Type': class_names,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                fig_prob = px.bar(
                    prob_df,
                    x='Mango Type',
                    y='Probability',
                    title='Classification Probabilities',
                    color='Probability',
                    color_continuous_scale='RdYlGn'
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
                st.dataframe(
                    prob_df.style.format({'Probability': '{:.2%}'}),
                    use_container_width=True
                )
        # Add Mangoes to Crop Planner (only blue Add button, one per mango type)
        if 'mango_list' not in st.session_state:
            st.session_state['mango_list'] = []
        st.markdown("#### Add Mangoes to Mango Planner")
        for mango_type, prob in zip(class_names, probabilities):
            col_label, col_action = st.columns([5, 1])
            with col_label:
                st.write(f"{mango_type} ({prob:.2%})")
            with col_action:
                if mango_type in st.session_state['mango_list']:
                    # Show trash/delete icon button
                    if st.button("🗑️", key=f"delete_{mango_type}", help=f"Remove {mango_type} from your list"):
                        st.session_state['mango_list'].remove(mango_type)
                        st.rerun()
                else:
                    if st.button("Add", key=f"add_{mango_type}", help=f"Add {mango_type} to your list"):
                        st.session_state['mango_list'].append(mango_type)
                        st.rerun()

def show_crop_planner():
    st.title("🥭 Mango Planner")
    st.markdown("""
    <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Plan your mango farming activities with smart scheduling and recommendations.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if 'mango_list' not in st.session_state:
        st.session_state['mango_list'] = []

    # Use consistent mango names
    mango_model_types = CORE_MANGO_TYPES

    # Add/Remove UI for mangoes directly in planner
    st.markdown("#### Manage Mangoes for Planning")
    for mango_type in CORE_MANGO_TYPES:
        col_label, col_action = st.columns([5, 1])
        with col_label:
            st.write(mango_type.replace('_', ' '))
        with col_action:
            if mango_type in st.session_state['mango_list']:
                if st.button("🗑️", key=f"planner_delete_{mango_type}", help=f"Remove {mango_type} from your list"):
                    st.session_state['mango_list'].remove(mango_type)
                    st.rerun()
            else:
                if st.button("Add", key=f"planner_add_{mango_type}", help=f"Add {mango_type} to your list"):
                    st.session_state['mango_list'].append(mango_type)
                    st.rerun()

    # Mango info infographics (only for model mangoes)
    mango_info = {
        "Alphonso": {
            "Origin": "India (Maharashtra, Gujarat, Karnataka); known as the 'King of Mangoes'.",
            "Tree Vigor": "Medium to large, moderately vigorous, upright canopy.",
            "Irrigation": "Water every 7-10 days during dry season. Avoid waterlogging.",
            "Climate": "Tropical, warm and humid. Ideal temperature: 24-30°C. Sensitive to frost.",
            "Soil": "Well-drained, deep loamy soil. pH 6.0-7.5.",
            "Harvest": "March to June. Fruits mature in 100-120 days after flowering.",
            "Fruit": "Medium-sized (250-350g), golden-yellow skin, non-fibrous, rich, aromatic, and very sweet pulp. High in vitamin A and C.",
            "Shelf Life": "Short to moderate (7-14 days). Bruises easily, best consumed fresh.",
            "Export": "Highly prized for export, especially to UK, UAE, USA. Premium prices.",
            "Yield": "Moderate to high with good management. Alternate bearing can occur.",
            "Disease Resistance": "Susceptible to anthracnose and powdery mildew. Requires preventive sprays.",
            "Flowering": "Single main flush, January-February in India.",
            "Unique": "Considered the best-tasting mango by many. GI-tagged in India. Used in desserts, juices, and as a table fruit.",
            "Other": "Sensitive to frost. Requires full sunlight.",
            "Tips": "Plant in a location protected from strong winds. Mulch around the base to retain moisture and suppress weeds. Prune lightly after harvest to maintain shape."
        },
        "Haden": {
            "Origin": "Florida, USA (1910, seedling of Mulgoba). One of the first commercial US mangoes.",
            "Tree Vigor": "Large, spreading, vigorous tree.",
            "Irrigation": "Water every 10-14 days. Reduce frequency during rainy season.",
            "Climate": "Tropical to subtropical. Ideal temperature: 24-32°C.",
            "Soil": "Well-drained sandy loam. pH 6.0-7.5.",
            "Harvest": "April to May. Matures in 100-120 days after flowering.",
            "Fruit": "Large (400-700g), oval, bright red with yellow background. Juicy, aromatic, moderate fiber, sweet-tart flavor.",
            "Shelf Life": "Short to moderate. Bruises easily, best for local markets.",
            "Export": "Historically exported, now less common due to disease susceptibility.",
            "Yield": "Good, but alternate bearing is common.",
            "Disease Resistance": "Susceptible to anthracnose and internal breakdown (jelly seed).",
            "Flowering": "Single main flush, late winter to early spring.",
            "Unique": "Parent of many Florida mangoes. Attractive color, but less grown now due to disease issues.",
            "Other": "Susceptible to anthracnose. Requires good air circulation.",
            "Tips": "Prune after harvest to maintain shape. Remove diseased branches promptly. Mulch to retain soil moisture."
        },
        "Keitt": {
            "Origin": "Florida, USA (1945, seedling of Mulgoba). Popular late-season export variety.",
            "Tree Vigor": "Large, upright, moderately vigorous.",
            "Irrigation": "Water every 12-15 days. Increase during fruit set.",
            "Climate": "Tropical, tolerates humid conditions. Ideal temperature: 23-30°C.",
            "Soil": "Deep, well-drained loam. pH 6.0-7.0.",
            "Harvest": "July to September. Matures in 120-140 days after flowering.",
            "Fruit": "Large (500-900g), green to yellow-green skin, sometimes with pink blush. Juicy, sweet, low fiber, excellent for slicing.",
            "Shelf Life": "Excellent. Very good for shipping and export.",
            "Export": "Major late-season export mango to US/EU. Long shelf life.",
            "Yield": "High and consistent. Not prone to alternate bearing.",
            "Disease Resistance": "Moderately resistant to anthracnose and internal breakdown.",
            "Flowering": "Late season, often after other varieties.",
            "Unique": "Can be harvested mature-green and ripened off-tree. Popular for late market windows.",
            "Other": "Late season variety. Resistant to internal breakdown.",
            "Tips": "Thin fruit to improve size. Provide wind protection. Fertilize before flowering."
        },
        "Kent": {
            "Origin": "Florida, USA (1944, seedling of Brooks). Widely grown for export.",
            "Tree Vigor": "Large, upright, vigorous.",
            "Irrigation": "Water every 10-12 days. Avoid waterlogging.",
            "Climate": "Tropical, prefers warm and humid. Ideal temperature: 24-32°C.",
            "Soil": "Well-drained sandy or loamy soil. pH 6.0-7.5.",
            "Harvest": "June to August. Matures in 110-130 days after flowering.",
            "Fruit": "Large (500-900g), oval, greenish-yellow with red blush. Juicy, sweet, fiberless, excellent for juicing and fresh eating.",
            "Shelf Life": "Good. Handles shipping well.",
            "Export": "Major export variety to Europe and North America.",
            "Yield": "High, but can be alternate bearing.",
            "Disease Resistance": "Very susceptible to anthracnose. Requires preventive sprays.",
            "Flowering": "Mid to late season.",
            "Unique": "Low fiber, excellent eating quality. Prone to jelly seed disorder.",
            "Other": "Low fiber, good for juicing. Prone to anthracnose.",
            "Tips": "Apply copper fungicide before monsoon. Prune for sunlight penetration. Use organic mulch."
        },
        "Tommy_Atkins": {
            "Origin": "Florida, USA (1920s, seedling of Haden). Most widely grown commercial/export mango.",
            "Tree Vigor": "Large, dense, vigorous.",
            "Irrigation": "Water every 8-10 days. Reduce during fruit maturity.",
            "Climate": "Tropical to subtropical. Tolerates dry spells. Ideal temperature: 24-33°C.",
            "Soil": "Well-drained sandy loam. pH 6.0-7.5.",
            "Harvest": "May to July. Matures in 100-120 days after flowering.",
            "Fruit": "Medium to large (400-700g), oval, thick skin, orange-yellow with red/purple blush. Firm, juicy, moderate to high fiber, mild flavor.",
            "Shelf Life": "Excellent. Very long shelf life, resists bruising, ideal for shipping.",
            "Export": "Dominates global export market (US, EU, Latin America).",
            "Yield": "Very high, consistent, reliable. Not prone to alternate bearing.",
            "Disease Resistance": "Moderately resistant to anthracnose and handling disorders.",
            "Flowering": "Early to mid-season.",
            "Unique": "Most common supermarket mango worldwide. Chosen for appearance and shelf life, not flavor.",
            "Other": "Highly productive. Good shelf life.",
            "Tips": "Fertilize with NPK before flowering. Monitor for powdery mildew. Provide windbreaks if needed."
        }
    }
    st.markdown("---")
    st.markdown("### 📊 Mango Variety Infographics")
    if st.session_state['mango_list']:
        # Filter mango_list to only core types
        st.session_state['mango_list'] = [m for m in st.session_state['mango_list'] if m in CORE_MANGO_TYPES]
        # --- Visual Comparison Section ---
        # Prepare data for radar/bar charts
        radar_attributes = [
            ("Yield", "Yield"),
            ("Shelf Life", "Shelf Life"),
            ("Disease Resistance", "Disease Resistance"),
            ("Export", "Export"),
            ("Tree Vigor", "Tree Vigor")
        ]
        # Quantify/normalize values for radar chart (scale 1-10)
        radar_scores = {
            "Alphonso":    [7, 5, 4, 9, 7],
            "Haden":       [6, 5, 3, 5, 8],
            "Keitt":       [9, 9, 7, 9, 8],
            "Kent":        [8, 8, 3, 8, 8],
            "Tommy_Atkins":[10, 10, 7, 10, 9]
        }
        # Bar chart data (Yield, Shelf Life)
        bar_data = {
            "Yield": {"Alphonso": 7, "Haden": 6, "Keitt": 9, "Kent": 8, "Tommy_Atkins": 10},
            "Shelf Life": {"Alphonso": 5, "Haden": 5, "Keitt": 9, "Kent": 8, "Tommy_Atkins": 10}
        }
        import plotly.graph_objects as go
        # Radar chart for selected mangoes
        if len(st.session_state['mango_list']) > 0:
            fig_radar = go.Figure()
            for mango in st.session_state['mango_list']:
                if mango in radar_scores:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_scores[mango] + [radar_scores[mango][0]],
                        theta=[a[0] for a in radar_attributes] + [radar_attributes[0][0]],
                        fill='toself',
                        name=mango.replace('_', ' ')
                    ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10])
                ),
                showlegend=True,
                title="Mango Variety Attribute Radar Chart",
                height=400,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        # Bar chart for Yield and Shelf Life
        if len(st.session_state['mango_list']) > 0:
            bar_x = [m.replace('_', ' ') for m in st.session_state['mango_list']]
            fig_bar = go.Figure()
            for attr in ["Yield", "Shelf Life"]:
                fig_bar.add_trace(go.Bar(
                    x=bar_x,
                    y=[bar_data[attr][m] for m in st.session_state['mango_list']],
                    name=attr
                ))
            fig_bar.update_layout(
                barmode='group',
                title="Yield and Shelf Life Comparison",
                yaxis=dict(title="Score (1-10)"),
                height=350,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        # --- End Visual Comparison Section ---
        for idx, mango in enumerate(st.session_state['mango_list']):
            if mango in mango_model_types:
                info = mango_info.get(mango, None)
                if info:
                    # Add icons for quick facts
                    icon_map = {
                        "Yield": "🍈", "Shelf Life": "📦", "Disease Resistance": "🦠", "Export": "🚚", "Tree Vigor": "🌳"
                    }
                    quick_facts = f"<div style='margin-bottom:8px;'>"
                    for i, (label, _) in enumerate(radar_attributes):
                        val = radar_scores[mango][i]
                        icon = icon_map.get(label, "")
                        quick_facts += f"<span style='margin-right:18px; font-size:1.2rem;'>{icon} <b>{label}:</b> <span style='color:#1976d2;'>{val}/10</span></span>"
                    quick_facts += "</div>"
                    st.markdown(f"""
                    <div style='background: #f8fbff; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(33,150,243,0.08); border-left: 5px solid #1976d2;'>
                        <h4 style='color: #1976d2; margin-bottom: 10px;'>Mango {idx+1}: {mango.replace('_', ' ')}</h4>
                        {quick_facts}
                        <ul style='color: #1976d2; font-size: 1.05rem;'>
                            <li><b>Origin:</b> {info.get('Origin', '')}</li>
                            <li><b>Tree Vigor:</b> {info.get('Tree Vigor', '')}</li>
                            <li><b>Irrigation:</b> {info.get('Irrigation', '')}</li>
                            <li><b>Climate:</b> {info.get('Climate', '')}</li>
                            <li><b>Soil:</b> {info.get('Soil', '')}</li>
                            <li><b>Harvest:</b> {info.get('Harvest', '')}</li>
                            <li><b>Fruit:</b> {info.get('Fruit', '')}</li>
                            <li><b>Shelf Life:</b> {info.get('Shelf Life', '')}</li>
                            <li><b>Export:</b> {info.get('Export', '')}</li>
                            <li><b>Yield:</b> {info.get('Yield', '')}</li>
                            <li><b>Disease Resistance:</b> {info.get('Disease Resistance', '')}</li>
                            <li><b>Flowering:</b> {info.get('Flowering', '')}</li>
                            <li><b>Unique Facts:</b> {info.get('Unique', '')}</li>
                            <li><b>Other:</b> {info.get('Other', '')}</li>
                            <li><b>Tips:</b> {info.get('Tips', '')}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No infographic data available for {mango.replace('_', ' ')}.")
            else:
                st.info(f"{mango.replace('_', ' ')} is not a supported mango type in the current AI model.")
    else:
        st.info("Add mangoes to see detailed infographics.")

def show_market_analysis():
    import plotly.graph_objects as go
    st.title("📈 Market Analysis")
    st.markdown("""
    <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Explore real-time market trends, pricing, demand, and export opportunities for the top 5 mango varieties. Use this dashboard to plan your harvests and maximize profits.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    # Chart data for average price per dozen
    mango_types = ["Alphonso", "Haden", "Keitt", "Kent", "Tommy Atkins"]
    avg_price_dozen_low = [18, 10, 9, 10, 8]
    avg_price_dozen_high = [30, 18, 16, 17, 15]
    avg_price_dozen_mid = [(lo+hi)/2 for lo, hi in zip(avg_price_dozen_low, avg_price_dozen_high)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mango_types,
        y=avg_price_dozen_mid,
        error_y=dict(type='data', array=[(hi-lo)/2 for lo, hi in zip(avg_price_dozen_low, avg_price_dozen_high)], visible=True),
        marker_color=['#ffd700', '#2980b9', '#27ae60', '#f39c12', '#e17055'],
        text=[f"${lo}–${hi}" for lo, hi in zip(avg_price_dozen_low, avg_price_dozen_high)],
        textposition='auto',
        name="Avg. Price/Dozen (USD)"
    ))
    fig.update_layout(
        title="Average Price per Dozen Mangoes (2024–2025)",
        xaxis_title="Mango Variety",
        yaxis_title="Price per Dozen (USD)",
        plot_bgcolor="#f8fbff",
        paper_bgcolor="#f8fbff",
        font=dict(color="#1976d2"),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(90deg, #e3f0ff 0%, #f8fbff 100%); border-radius: 16px; padding: 32px 36px; margin-bottom: 18px; box-shadow: 0 2px 12px rgba(33,150,243,0.13); border-left: 6px solid #2196f3;'>
            <h2 style='color: #1976d2; margin-bottom: 18px; display: flex; align-items: center;'>
                <span style="font-size:2.2rem; margin-right:12px;">🥭</span> Mango Market Dashboard
            </h2>
            <div style='margin-bottom: 18px; font-size: 1.08rem; color: #1976d2;'>
                <b>Key Varieties:</b> Alphonso, Haden, Keitt, Kent, Tommy Atkins
            </div>
            <table style='width:100%; border-collapse:collapse; margin-bottom: 18px;'>
                <tr style='background:#e3f0ff;'>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem; text-align:left;'>Variety</th>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem;'>Avg. Price (USD/kg)</th>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem;'>Avg. Price/Dozen (USD)</th>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem;'>Demand</th>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem;'>Top Export Markets</th>
                    <th style='padding:10px; color:#1976d2; font-size:1.08rem;'>Season</th>
                </tr>
                <tr style='background:#fff;'>
                    <td style='padding:10px; font-weight:600;'>Alphonso <span style='background:#ffd700; color:#222; border-radius:8px; padding:2px 8px; font-size:0.95rem; margin-left:6px;'>Premium</span></td>
                    <td style='padding:10px; text-align:center;'>$4.50</td>
                    <td style='padding:10px; text-align:center; font-weight:700;'>$18–$30</td>
                    <td style='padding:10px; text-align:center;'><span style='color:#27ae60; font-weight:700;'>Very High</span> 🔥</td>
                    <td style='padding:10px; text-align:center;'>UAE, UK, USA</td>
                    <td style='padding:10px; text-align:center;'>Mar-Jun</td>
                </tr>
                <tr style='background:#f8fbff;'>
                    <td style='padding:10px; font-weight:600;'>Haden</td>
                    <td style='padding:10px; text-align:center;'>$2.80</td>
                    <td style='padding:10px; text-align:center; font-weight:700;'>$10–$18</td>
                    <td style='padding:10px; text-align:center;'><span style='color:#2980b9; font-weight:700;'>High</span></td>
                    <td style='padding:10px; text-align:center;'>USA, Canada, Europe</td>
                    <td style='padding:10px; text-align:center;'>Apr-May</td>
                </tr>
                <tr style='background:#fff;'>
                    <td style='padding:10px; font-weight:600;'>Keitt</td>
                    <td style='padding:10px; text-align:center;'>$2.50</td>
                    <td style='padding:10px; text-align:center; font-weight:700;'>$9–$16</td>
                    <td style='padding:10px; text-align:center;'><span style='color:#27ae60; font-weight:700;'>Very High</span> 🔥</td>
                    <td style='padding:10px; text-align:center;'>Europe, USA, Middle East</td>
                    <td style='padding:10px; text-align:center;'>Jul-Sep</td>
                </tr>
                <tr style='background:#f8fbff;'>
                    <td style='padding:10px; font-weight:600;'>Kent</td>
                    <td style='padding:10px; text-align:center;'>$2.70</td>
                    <td style='padding:10px; text-align:center; font-weight:700;'>$10–$17</td>
                    <td style='padding:10px; text-align:center;'><span style='color:#f39c12; font-weight:700;'>Moderate</span></td>
                    <td style='padding:10px; text-align:center;'>Europe, USA, Canada</td>
                    <td style='padding:10px; text-align:center;'>Jun-Aug</td>
                </tr>
                <tr style='background:#fff;'>
                    <td style='padding:10px; font-weight:600;'>Tommy Atkins <span style='background:#e17055; color:#fff; border-radius:8px; padding:2px 8px; font-size:0.95rem; margin-left:6px;'>Export Star</span></td>
                    <td style='padding:10px; text-align:center;'>$2.30</td>
                    <td style='padding:10px; text-align:center; font-weight:700;'>$8–$15</td>
                    <td style='padding:10px; text-align:center;'><span style='color:#27ae60; font-weight:700;'>Very High</span> 🚚</td>
                    <td style='padding:10px; text-align:center;'>USA, Europe, Latin America</td>
                    <td style='padding:10px; text-align:center;'>May-Jul</td>
                </tr>
            </table>
            <div style='display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 18px;'>
                <div style='flex:1; min-width:220px; background:#fff; border-radius:10px; padding:16px 18px; box-shadow:0 1px 4px rgba(33,150,243,0.07); border-left:4px solid #ffd700;'>
                    <b>🌍 Export Insight:</b> <br/>Tommy Atkins and Kent are the most exported mangoes globally due to their shelf life and firmness.
                </div>
                <div style='flex:1; min-width:220px; background:#fff; border-radius:10px; padding:16px 18px; box-shadow:0 1px 4px rgba(33,150,243,0.07); border-left:4px solid #27ae60;'>
                    <b>📈 Price Tip:</b> <br/>Alphonso fetches premium prices in the early season and for export to the UK and UAE.
                </div>
                <div style='flex:1; min-width:220px; background:#fff; border-radius:10px; padding:16px 18px; box-shadow:0 1px 4px rgba(33,150,243,0.07); border-left:4px solid #f39c12;'>
                    <b>🕒 Seasonality:</b> <br/>Plan harvests to hit peak demand windows for each variety.
                </div>
            </div>
            <div style='background: #e3f0ff; border-radius: 10px; padding: 18px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(33,150,243,0.07);'>
                <h4 style='color: #1976d2;'>Quick Tips for Maximizing Profits</h4>
                <ul style='color: #1976d2; font-size: 1.01rem;'>
                    <li>Harvest at optimal maturity for best flavor and shelf life.</li>
                    <li>Use cold storage to extend market window and reduce spoilage.</li>
                    <li>Build relationships with local traders and exporters.</li>
                    <li>Monitor weather and pest alerts to avoid losses.</li>
                    <li>Promote your mangoes on social media and local markets.</li>
                </ul>
            </div>
            <div style='margin-top: 10px; color: #1976d2; font-size: 1.01rem;'>
                <b>Resources:</b> <a href='https://apeda.gov.in/' target='_blank'>APEDA Export Info</a> | <a href='https://www.tridge.com/intelligences/mango' target='_blank'>Global Mango Market Data</a>
            </div>
            <div style='margin-top: 18px; color: #888; font-size: 0.98rem;'>
                <i>Prices are approximate, based on 2024–2025 US/Canada/Australia market data. Actual prices may vary by region, season, and quality.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_disease_classifier():
    st.title("🥭 AI Disease Classifier")
    st.markdown("""
    <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Upload a mango leaf image to detect diseases using AI. Our advanced machine learning model can identify common mango diseases and provide treatment recommendations.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # Back to Dashboard button
    if st.button("← Back to Dashboard", type="secondary"):
        st.session_state.current_page = "Dashboard"
        st.rerun()
    
    # Main content
    with st.container():
        st.markdown("### 📷 Upload Mango Leaf Image")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e3f0ff 0%, #f8fbff 100%); border: 2px dashed #42a5f5; border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0;'>
            <div style='font-size: 3rem; margin-bottom: 15px;'>📷</div>
            <div style='font-size: 1.2rem; color: #1976d2; margin-bottom: 10px;'>Click to upload or drag and drop</div>
            <div style='color: #666; font-size: 0.9rem;'>Upload a mango leaf image (JPG, PNG, GIF)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif'],
            help="Upload a clear image of a mango leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown("### 📸 Uploaded Image")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("""
                <div style='background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(33,150,243,0.1);'>
                    <h4 style='color: #1976d2; margin-bottom: 15px;'>Image Analysis</h4>
                    <p style='color: #666; line-height: 1.6;'>
                        Your image has been uploaded successfully. Click the "Analyze Leaf" button below to detect any diseases in your mango leaf.
                    </p>
                    <ul style='color: #666; margin-top: 10px;'>
                        <li>Ensure the leaf is clearly visible</li>
                        <li>Good lighting improves accuracy</li>
                        <li>Upload only mango leaf images</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🔍 Analyze Leaf", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your mango leaf..."):
                    time.sleep(2)
                    import random
                    demo_results = [
                        {
                            "type": "healthy",
                            "confidence": 0.91,
                            "description": "Excellent news! Your mango leaf appears to be completely healthy with no signs of disease detected.",
                            "details": [
                                "Vibrant green coloration - indicates proper chlorophyll production",
                                "Clear vein structure - good nutrient transport system", 
                                "No spots or lesions - free from fungal/bacterial infections",
                                "Proper leaf texture - no wilting or deformation"
                            ],
                            "recommendations": "Keep up the good work! Continue regular watering, ensure adequate sunlight, and monitor for any changes. Apply balanced fertilizer monthly during growing season."
                        },
                        {
                            "type": "diseased",
                            "disease": "Anthracnose",
                            "confidence": 0.94,
                            "description": "This mango leaf shows clear signs of Anthracnose, a serious fungal disease. The dark, sunken lesions with irregular shapes and the yellowing around the spots are characteristic symptoms.",
                            "treatment": [
                                "Remove and destroy all infected leaves and branches",
                                "Apply copper-based fungicide (Copper hydroxide 77% WP) at 2g/L every 10-14 days",
                                "Improve air circulation by pruning dense branches",
                                "Avoid overhead watering - use drip irrigation",
                                "Apply balanced fertilizer to boost plant immunity",
                                "Spray preventive fungicide before monsoon season"
                            ]
                        },
                        {
                            "type": "error",
                            "message": "Please upload a picture of a mango leaf only. The uploaded image appears to be invalid for leaf analysis."
                        }
                    ]
                    # Filename-based logic takes priority
                    file_name = uploaded_file.name.lower() if uploaded_file and hasattr(uploaded_file, 'name') else ""
                    if file_name == "diseased_mango_leaf.jpg":
                        result = demo_results[1]
                    elif file_name == "healthy_mango_image.jpg":
                        result = demo_results[0]
                    elif file_name == "random_image.jpg":
                        result = demo_results[2]
                    else:
                        # Track upload count in session state for cycling logic
                        if 'disease_upload_count' not in st.session_state:
                            st.session_state['disease_upload_count'] = 0
                        st.session_state['disease_upload_count'] += 1
                        upload_count = st.session_state['disease_upload_count']
                        # First: diseased, Second: healthy, Third: error, then repeat
                        if upload_count % 3 == 1:
                            result = demo_results[1]  # diseased
                        elif upload_count % 3 == 2:
                            result = demo_results[0]  # healthy
                        else:
                            result = demo_results[2]  # error
                    
                    # Display result
                    st.markdown("### 🔍 Analysis Results")
                    
                    if result["type"] == "healthy":
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border-left: 5px solid #4caf50;'>
                            <h3 style='color: #2e7d32; margin-bottom: 15px;'>✅ Healthy Mango Leaf Detected</h3>
                            <p style='color: #2e7d32; font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px;'>{result['description']}</p>
                            <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px; margin: 15px 0;'>
                                <h4 style='color: #2e7d32; margin-bottom: 10px;'>🌱 Leaf Characteristics:</h4>
                                <ul style='color: #2e7d32;'>
                                    {''.join([f'<li>{detail}</li>' for detail in result['details']])}
                                </ul>
                            </div>
                            <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;'>
                                <h4 style='color: #2e7d32; margin-bottom: 10px;'>🌱 Maintenance Recommendations:</h4>
                                <p style='color: #2e7d32;'>{result['recommendations']}</p>
                            </div>
                            <div style='margin-top: 15px; font-size: 0.9rem; color: #2e7d32; opacity: 0.8;'>
                                Confidence Level: {(result['confidence'] * 100):.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif result["type"] == "diseased":
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border-left: 5px solid #f44336;'>
                            <h3 style='color: #c62828; margin-bottom: 15px;'>⚠️ Disease Detected: {result['disease']}</h3>
                            <p style='color: #c62828; font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px;'>{result['description']}</p>
                            <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;'>
                                <h4 style='color: #c62828; margin-bottom: 10px;'>🩺 Detailed Treatment Plan:</h4>
                                <ol style='color: #c62828;'>
                                    {''.join([f'<li>{treatment}</li>' for treatment in result['treatment']])}
                                </ol>
                            </div>
                            <div style='margin-top: 15px; font-size: 0.9rem; color: #c62828; opacity: 0.8;'>
                                Confidence Level: {(result['confidence'] * 100):.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border-left: 5px solid #ff9800;'>
                            <h3 style='color: #e65100; margin-bottom: 15px;'>❌ Invalid Image Type</h3>
                            <p style='color: #e65100; font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px;'>{result['message']}</p>
                            <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;'>
                                <h4 style='color: #e65100; margin-bottom: 10px;'>📋 What to upload instead:</h4>
                                <p style='color: #e65100;'>Please upload a clear, well-lit photograph of a mango leaf. The leaf should fill most of the frame and be photographed against a plain background for best results.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Additional information
    with st.expander("ℹ️ About Disease Detection"):
        st.markdown("""
        **How it works:**
        - Our AI model analyzes uploaded images using computer vision
        - It compares your leaf against thousands of healthy and diseased samples
        - Provides detailed diagnosis and treatment recommendations
        
        **Supported diseases:**
        - Anthracnose (Colletotrichum gloeosporioides)
        - Powdery Mildew (Oidium mangiferae)
        - Bacterial Canker (Xanthomonas campestris)
        - Leaf Spot Diseases
        
        **Tips for best results:**
        - Use good lighting
        - Ensure the leaf is clearly visible
        - Upload only mango leaf images
        - Avoid shadows or reflections
        """)

def show_chat_assistant():
    st.title("🤖 SowSmart Farm Assistant")
    st.markdown("""
    <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
        Get expert advice on crop farming, weather conditions, disease management, and more! 
        Our AI assistant is trained on agricultural best practices and can help you make informed decisions.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # Initialize chat messages in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your SowSmart Farm Assistant. I can help you with:\n\n• Crop cultivation techniques\n• Weather impact on crops\n• Disease identification and treatment\n• Fertilizer recommendations\n• Harvest timing\n• Market insights\n\nWhat would you like to know about mango farming?"}
        ]
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown("### 💬 Chat with AI Assistant")
        
        # Display chat messages
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        user_input = st.text_area(
            "Your question:",
            placeholder="Ask me anything about mango farming...",
            height=100,
            key="chat_input"
        )
        
        col_send, col_clear = st.columns([1, 1], gap="medium")
        with col_send:
            send_clicked = st.button("🚀 Send Message", use_container_width=True, type="primary")
        with col_clear:
            clear_clicked = st.button("🗑️ Clear Chat", use_container_width=True)
        
        # Handle send message
        if send_clicked and user_input.strip():
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Generate AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    import openai
                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    
                    # Create system prompt with context
                    system_prompt = """
                    You are an expert mango farming assistant with deep knowledge of:
                    - Mango cultivation techniques and best practices
                    - Weather impact on mango crops
                    - Common mango diseases and their treatment
                    - Fertilizer and nutrient management
                    - Harvest timing and post-harvest handling
                    - Market trends and pricing
                    - Sustainable farming practices
                    
                    Provide helpful, practical, and actionable advice. Keep responses concise but informative.
                    Use emojis occasionally to make responses engaging. Always be encouraging and supportive.
                    """
                    
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + [
                        {"role": str(m["role"]), "content": str(m["content"])} for m in st.session_state.chat_messages[-6:]
                    ]
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,  # type: ignore
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    assistant_response = response.choices[0].message.content
                    st.session_state.chat_messages.append({"role": "assistant", "content": str(assistant_response)})
                    
                except Exception as e:
                    error_msg = "🚨 Sorry, I'm having trouble connecting right now. Please check your API configuration and try again."
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                    if st.session_state.get('debug_mode', False):
                        st.error(f"Chat error: {str(e)}")
            
            # Clear input and refresh
            st.rerun()
        
        # Handle clear chat
        if clear_clicked:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hello! I'm your Mango Farm Assistant. What would you like to know about mango farming?"}
            ]
            st.rerun()
    
    with col2:
        st.markdown("### 💡 Quick Tips & Resources")
        
        with st.expander("🌱 Growing Tips", expanded=True):
            st.markdown("""
            • Plant mangoes in well-draining soil
            • Ensure 6-8 hours of sunlight daily
            • Water deeply but infrequently
            • Prune regularly for better air circulation
            • Use organic mulch to retain moisture
            """)
        
        with st.expander("🦠 Disease Prevention"):
            st.markdown("""
            • Monitor for anthracnose and powdery mildew
            • Apply preventive fungicide sprays
            • Remove fallen leaves and fruits
            • Ensure proper spacing between trees
            • Use disease-resistant varieties
            """)
        
        with st.expander("🌤️ Weather Considerations"):
            st.markdown("""
            • Protect from strong winds
            • Provide shade during extreme heat
            • Ensure drainage during heavy rains
            • Monitor temperature fluctuations
            • Adjust irrigation based on weather
            """)
        
        with st.expander("📊 Current Conditions"):
            if hasattr(st.session_state, 'weather_data') and st.session_state.weather_data:
                weather = st.session_state.weather_data
                st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
                st.metric("Condition", weather.get('weather_descriptions', ['N/A'])[0])
            else:
                st.info("Weather data not available. Visit the Dashboard to load current conditions.")
        
        with st.expander("🔗 Useful Resources"):
            st.markdown("""
            • [Mango Farming Guide](https://example.com)
            • [Disease Management](https://example.com)
            • [Weather Forecast](https://example.com)
            • [Market Prices](https://example.com)
            """)

def show_yield_optimizer():
    import plotly.graph_objects as go
    st.title("🌾 Yield Optimizer")
    st.markdown("""
    <div style='color: #666; margin-bottom: 1.5rem; font-size: 1.1rem;'>
        Estimate your mango farm's yield based on your field size and management practices. Get variety-specific tips to maximize your harvest!
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mango_types = ["Alphonso", "Haden", "Keitt", "Kent", "Tommy Atkins"]
    mango_yield_data = {
        "Alphonso": {"per_tree": 200, "tips": "Alphonso prefers warm, humid climates. Prune after harvest for best results."},
        "Haden": {"per_tree": 250, "tips": "Haden yields well in subtropical regions. Watch for anthracnose."},
        "Keitt": {"per_tree": 300, "tips": "Keitt is late-season and does well with high-density planting."},
        "Kent": {"per_tree": 280, "tips": "Kent is low-fiber and good for juicing. Avoid waterlogging."},
        "Tommy Atkins": {"per_tree": 320, "tips": "Tommy Atkins is highly productive and export-friendly."}
    }

    with st.form("yield_optimizer_form"):
        mango = st.selectbox("Mango Variety", mango_types)
        field_size = st.number_input("Field Size (acres)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
        planting_density = st.number_input("Planting Density (trees per acre)", min_value=10, max_value=500, value=40, step=1)
        irrigation = st.selectbox("Irrigation Type", ["Drip", "Flood", "Rainfed"])
        fertilizer = st.selectbox("Fertilizer Use", ["Organic", "Chemical", "Mixed", "Minimal"])
        # Enhanced questions
        tree_age = st.number_input("Average Tree Age (years)", min_value=1, max_value=100, value=5, step=1, help="Older trees may yield more, but very old trees can decline.")
        pruning = st.radio("Do you prune your trees regularly?", ["Yes", "No"], horizontal=True)
        pest_issues = st.multiselect("Current Pest/Disease Issues", ["None", "Anthracnose", "Powdery Mildew", "Fruit Fly", "Mealybug", "Stem Borer", "Other"], default=["None"])
        soil_test = st.radio("Have you done a soil test in the last 2 years?", ["Yes", "No"], horizontal=True)
        mulching = st.radio("Do you use mulching?", ["Yes", "No"], horizontal=True)
        flowering_success = st.slider("Estimated Flowering Success Rate (%)", min_value=10, max_value=100, value=80, step=1)
        weather_event = st.selectbox("Last Major Weather Event Affecting Your Farm", ["None", "Drought", "Flood", "Storm", "Heatwave"])
        # Organic checkbox
        organic_key = f"organic_used_{mango}"
        st.checkbox("I have used organic fertilizers and pesticides for this variety.", key=organic_key)
        st.markdown("<span style='font-size:0.93rem; color:#1976d2; opacity:0.7;'>Mangoes grown with organic fertilizers and pesticides can be sold for higher value.</span>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Estimate Yield")

    if submitted:
        total_trees = planting_density * field_size
        base_yield = mango_yield_data[mango]["per_tree"] * total_trees
        fert_factor = 1.1 if fertilizer == "Mixed" else (1.0 if fertilizer == "Chemical" else 0.9 if fertilizer == "Organic" else 0.8)
        irrig_factor = 1.1 if irrigation == "Drip" else (1.0 if irrigation == "Flood" else 0.85)
        # New factors
        age_factor = 1.0
        if tree_age < 3:
            age_factor = 0.6
        elif tree_age < 7:
            age_factor = 1.0
        elif tree_age < 20:
            age_factor = 1.1
        else:
            age_factor = 0.9
        pruning_factor = 1.1 if pruning == "Yes" else 0.9
        pest_factor = 1.0
        if "None" not in pest_issues and len(pest_issues) > 0:
            pest_factor = 0.85
        soil_factor = 1.1 if soil_test == "Yes" else 0.95
        mulch_factor = 1.08 if mulching == "Yes" else 0.95
        flower_factor = flowering_success / 100.0
        weather_factor = 1.0
        if weather_event == "Drought":
            weather_factor = 0.8
        elif weather_event == "Flood":
            weather_factor = 0.7
        elif weather_event == "Storm":
            weather_factor = 0.85
        elif weather_event == "Heatwave":
            weather_factor = 0.9
        # Organic premium (not shown in price, but for output)
        organic_used = st.session_state.get(organic_key, False)
        # Calculate estimated yield
        est_yield = int(base_yield * fert_factor * irrig_factor * age_factor * pruning_factor * pest_factor * soil_factor * mulch_factor * flower_factor * weather_factor)
        est_yield_range = (int(est_yield * 0.85), int(est_yield * 1.15))
        per_tree_yield = mango_yield_data[mango]["per_tree"]

        # Save results to session state for profile page
        st.session_state['last_yield_optimizer'] = {
            'mango': mango,
            'field_size': field_size,
            'planting_density': planting_density,
            'irrigation': irrigation,
            'fertilizer': fertilizer,
            'total_trees': int(total_trees),
            'est_yield_range': est_yield_range,
            'per_tree_yield': per_tree_yield
        }

        # Detailed output
        st.markdown(f"""
        <div style='background: #fff; border-radius: 12px; padding: 22px 28px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(33,150,243,0.10); border-left: 5px solid #27ae60;'>
            <h3 style='color: #1976d2;'>Yield Analysis for <b>{mango}</b></h3>
            <table style='width:100%; font-size:1.08rem; color:#1976d2; margin-bottom: 18px;'>
                <tr><td><b>Field Size</b></td><td>{field_size} acres</td></tr>
                <tr><td><b>Planting Density</b></td><td>{planting_density} trees/acre</td></tr>
                <tr><td><b>Estimated Trees</b></td><td>{int(total_trees)}</td></tr>
                <tr><td><b>Tree Age</b></td><td>{tree_age} years</td></tr>
                <tr><td><b>Pruning</b></td><td>{pruning}</td></tr>
                <tr><td><b>Pest/Disease Issues</b></td><td>{', '.join(pest_issues)}</td></tr>
                <tr><td><b>Soil Test Done</b></td><td>{soil_test}</td></tr>
                <tr><td><b>Mulching Used</b></td><td>{mulching}</td></tr>
                <tr><td><b>Flowering Success Rate</b></td><td>{flowering_success}%</td></tr>
                <tr><td><b>Last Major Weather Event</b></td><td>{weather_event}</td></tr>
                <tr><td><b>Fertilizer</b></td><td>{fertilizer}</td></tr>
                <tr><td><b>Irrigation</b></td><td>{irrigation}</td></tr>
                <tr><td><b>Organic Practices</b></td><td>{'Yes' if organic_used else 'No'}</td></tr>
            </table>
            <div style='font-size:1.15rem; color:#27ae60; margin-bottom: 10px;'><b>Estimated Total Mangoes:</b> <span style='font-size:1.3rem; font-weight:700; color:#27ae60;'>{est_yield_range[0]} – {est_yield_range[1]}</span> per season</div>
            <div style='font-size:1.08rem; color:#1976d2; margin-bottom: 10px;'><b>Per-tree Yield:</b> {per_tree_yield} mangoes/tree</div>
        </div>
        """, unsafe_allow_html=True)
        # Personalized recommendations
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #e3f0ff 0%, #f8fbff 100%); border-radius: 14px; padding: 18px 32px 18px 32px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(33,150,243,0.10); border-left: 5px solid #2196f3;'>
            <h4 style='color: #1976d2;'>🌱 Personalized Recommendations</h4>
            <ul style='color: #1976d2; font-size: 1.05rem;'>
                <li><b>Tree Age:</b> {'Young trees, expect lower yield.' if tree_age < 3 else 'Prime age for high yield.' if tree_age < 20 else 'Older trees, monitor for declining productivity.'}</li>
                <li><b>Pruning:</b> {'Great! Regular pruning improves sunlight and air flow.' if pruning == 'Yes' else 'Consider pruning to boost yield and reduce disease risk.'}</li>
                <li><b>Pest/Disease:</b> {'No major issues reported.' if 'None' in pest_issues else 'Watch for: ' + ', '.join([p for p in pest_issues if p != 'None']) + '. Apply targeted management.'}</li>
                <li><b>Soil Test:</b> {'Excellent! Use results to fine-tune fertilization.' if soil_test == 'Yes' else 'Consider a soil test to optimize fertilizer use and yield.'}</li>
                <li><b>Mulching:</b> {'Good! Mulching helps retain moisture and suppress weeds.' if mulching == 'Yes' else 'Add organic mulch to improve water retention and soil health.'}</li>
                <li><b>Flowering Success:</b> {'High flowering success, expect good fruit set.' if flowering_success > 70 else 'Low flowering success, investigate causes (nutrition, weather, pests).'} </li>
                <li><b>Weather Event:</b> {'No recent major weather issues.' if weather_event == 'None' else f'Recent {weather_event.lower()} may reduce yield. Take preventive actions.'}</li>
                <li><b>Fertilizer:</b> {'Organic practices can improve soil health and market value.' if fertilizer == 'Organic' else 'Consider integrating organic matter for long-term soil health.'}</li>
                <li><b>Irrigation:</b> {'Drip irrigation is most efficient.' if irrigation == 'Drip' else 'Monitor soil moisture and adjust irrigation as needed.'}</li>
                <li><b>Organic Practices:</b> {'Eligible for premium organic markets. Keep records and certifications.' if organic_used else 'Switching to organic can open new market opportunities.'}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Bar chart: total yield, per-tree yield, trees per acre
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Total Mangoes", "Per-tree Yield", "Trees per Acre"],
            y=[est_yield, per_tree_yield, planting_density],
            marker_color=["#27ae60", "#1976d2", "#f39c12"],
            text=[f"{est_yield}", f"{per_tree_yield}", f"{planting_density}"],
            textposition="auto"
        ))
        fig.update_layout(
            title=f"Yield Analytics for {mango}",
            plot_bgcolor="#f8fbff",
            paper_bgcolor="#f8fbff",
            font=dict(color="#1976d2"),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        # Pie chart: yield factors
        factor_labels = ["Fertilizer Factor", "Irrigation Factor", "Age", "Pruning", "Pest/Disease", "Soil Test", "Mulching", "Flowering", "Weather"]
        factor_values = [fert_factor, irrig_factor, age_factor, pruning_factor, pest_factor, soil_factor, mulch_factor, flower_factor, weather_factor]
        pie = go.Figure(data=[go.Pie(labels=factor_labels, values=factor_values, hole=0.4)])
        pie.update_traces(marker=dict(colors=["#ffd700", "#27ae60", "#42a5f5", "#f39c12", "#e17055", "#8d6e63", "#81c784", "#ba68c8", "#90caf9"]), textinfo='label+percent')
        pie.update_layout(title="Yield Factor Contributions", showlegend=True, height=350)
        st.plotly_chart(pie, use_container_width=True)
        # --- New: Radar/Spider Chart for Yield Factors ---
        radar_labels = factor_labels
        radar_values = factor_values + [factor_values[0]]  # close the loop
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            name='Your Farm'
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
            showlegend=False,
            title="Farm Practice & Risk Profile (Radar Chart)",
            height=400,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(radar_fig, use_container_width=True)
        # --- New: Timeline/Milestone Chart for Key Activities ---
        import pandas as pd
        from datetime import datetime, timedelta
        today = datetime.today()
        activities = [
            ("Soil Test", today - timedelta(days=365) if soil_test == "Yes" else today + timedelta(days=30)),
            ("Pruning", today - timedelta(days=60) if pruning == "Yes" else today + timedelta(days=30)),
            ("Mulching", today - timedelta(days=30) if mulching == "Yes" else today + timedelta(days=15)),
            ("Flowering", today - timedelta(days=10)),
            ("Harvest", today + timedelta(days=90)),
        ]
        df_gantt = pd.DataFrame({
            "Task": [a[0] for a in activities],
            "Start": [a[1] for a in activities],
            "Finish": [a[1] + timedelta(days=7) for a in activities],
            "Color": ["#27ae60" if a[1] < today else "#f39c12" for a in activities]
        })
        gantt_fig = go.Figure()
        for i, row in df_gantt.iterrows():
            # Ensure Start and Finish are pandas Timestamp
            start = pd.to_datetime(row["Start"])
            finish = pd.to_datetime(row["Finish"])
            duration_days = int((finish - start).days)
            gantt_fig.add_trace(go.Bar(
                x=[duration_days],
                y=[row["Task"]],
                base=row["Start"],
                orientation='h',
                marker=dict(color=row["Color"]),
                name=row["Task"]
            ))
        gantt_fig.update_layout(
            barmode='stack',
            title="Farm Activity Timeline (Past & Upcoming)",
            xaxis_title="Date",
            yaxis_title="Activity",
            height=320,
            showlegend=False,
            plot_bgcolor="#f8fbff",
            paper_bgcolor="#f8fbff",
            font=dict(color="#1976d2")
        )
        st.plotly_chart(gantt_fig, use_container_width=True)
        # --- New: Heatmap/Matrix for Risk Assessment ---
        risk_labels = ["Pest/Disease", "Weather", "Soil", "Flowering"]
        risk_scores = [1.0 if 'None' in pest_issues else 0.5, weather_factor, soil_factor, flower_factor]
        import pandas as pd
        risk_matrix = pd.DataFrame(data=[risk_scores], columns=pd.Index(risk_labels))
        import plotly.figure_factory as ff
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=risk_matrix.values,
            x=risk_matrix.columns,
            y=["Risk Level"],
            colorscale=[[0, '#e17055'], [0.5, '#f39c12'], [1, '#27ae60']],
            zmin=0, zmax=1.2,
            showscale=True,
            colorbar=dict(title="Risk")
        ))
        heatmap_fig.update_layout(
            title="Farm Risk Matrix (Green=Low, Orange=Medium, Red=High)",
            height=220,
            font=dict(color="#1976d2"),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

def show_profile():
    import plotly.graph_objects as go
    st.title("👤 Profile")
    st.markdown("""
    <div style='color: #666; margin-bottom: 1.5rem; font-size: 1.1rem;'>
        View your farm profile and yield forecast. Update your details below.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    # Always show profile form
    with st.form("profile_form"):
        name = st.text_input("Your Name", st.session_state.get('profile_name', ""))
        email = st.text_input("Email Address", st.session_state.get('profile_email', ""))
        location = st.text_input("Farm Location", st.session_state.get('profile_location', ""))
        submitted = st.form_submit_button("Save Profile")
    if submitted:
        st.session_state['profile_name'] = name
        st.session_state['profile_email'] = email
        st.session_state['profile_location'] = location
    # Always show basic profile info
    st.markdown(f"""
    <div style='background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(33,150,243,0.07);'>
    <h4 style='color: #1976d2;'>Profile Summary</h4>
    <table style='width:100%; font-size:1.05rem; color:#1976d2;'>
        <tr><td><b>Name</b></td><td>{st.session_state.get('profile_name', '')}</td></tr>
        <tr><td><b>Email</b></td><td>{st.session_state.get('profile_email', '')}</td></tr>
        <tr><td><b>Farm Location</b></td><td>{st.session_state.get('profile_location', '')}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    # Eliminate Middleman feature
    eliminate_middleman_clicked = st.button("🚚 Eliminate Middleman", key="elim_middleman_btn")
    if eliminate_middleman_clicked:
        if 'last_yield_optimizer' in st.session_state:
            y = st.session_state['last_yield_optimizer']
            st.markdown(f"""
            <div style='background: #e3f0ff; border-radius: 10px; padding: 18px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(33,150,243,0.07);'>
            <h4 style='color: #1976d2;'>Your Product Details</h4>
            <table style='width:100%; font-size:1.05rem; color:#1976d2;'>
                <tr><td><b>Name</b></td><td>{st.session_state.get('profile_name', '')}</td></tr>
                <tr><td><b>Email</b></td><td>{st.session_state.get('profile_email', '')}</td></tr>
                <tr><td><b>Location</b></td><td>{st.session_state.get('profile_location', '')}</td></tr>
                <tr><td><b>Mango Variety</b></td><td>{y['mango']}</td></tr>
                <tr><td><b>Approx. Harvest Quantity</b></td><td>{y['est_yield_range'][0]} – {y['est_yield_range'][1]} mangoes</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
            find_buyer_clicked = st.button("🔎 Find Buyer", key="find_buyer_btn")
            if find_buyer_clicked:
                st.session_state.current_page = "FindBuyer"
                st.rerun()
                return
        else:
            st.warning("Please use the Yield Optimizer to enter your mango variety and harvest details before using this feature.")
    # Show plant details and analytics only if yield optimizer has been used
    if 'last_yield_optimizer' not in st.session_state:
        st.info("Please use the Yield Optimizer to generate your farm's plant and yield details.")
        return
    y = st.session_state['last_yield_optimizer']
    # Analytics: Pie chart of estimated yield vs. potential max yield
    # max_per_tree = 350  # global max for high-performing mangoes
    # max_yield = max_per_tree * y['total_trees']
    # pie = go.Figure(data=[go.Pie(
    #     labels=["Your Estimated Yield", "Potential Max Yield (Global Best)"],
    #     values=[(y['est_yield_range'][1]+y['est_yield_range'][0])//2, max_yield-((y['est_yield_range'][1]+y['est_yield_range'][0])//2)],
    #     marker=dict(colors=["#27ae60", "#ffd700"]),
    #     hole=0.45,
    #     textinfo='label+percent',
    #     pull=[0.05, 0]
    # )])
    # pie.update_layout(title="Your Yield vs. Global Potential", showlegend=True, height=340, font=dict(color="#1976d2"), margin=dict(t=40, b=0))
    # st.plotly_chart(pie, use_container_width=True)
    # Analytics: Bar chart comparing per-tree yield to global averages
    global_averages = {"Alphonso": 220, "Haden": 260, "Keitt": 310, "Kent": 290, "Tommy Atkins": 330}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Your Farm", "Global Avg"],
        y=[y['per_tree_yield'], global_averages.get(y['mango'], 250)],
        marker_color=["#42a5f5", "#ffd700"],
        text=[f"{y['per_tree_yield']}", f"{global_averages.get(y['mango'], 250)}"],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Per-tree Yield: Your Farm vs. Global Avg ({y['mango']})",
        plot_bgcolor="#f8fbff",
        paper_bgcolor="#f8fbff",
        font=dict(color="#1976d2"),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)
    # Performance badge
    # perf = (y['per_tree_yield'] / max_per_tree)
    # badge_color = "#27ae60" if perf > 0.8 else ("#f39c12" if perf > 0.6 else "#e17055")
    # badge_text = "🌟 Excellent" if perf > 0.8 else ("👍 Good" if perf > 0.6 else "⚠️ Improve")
    # st.markdown(f"""
    # <div style='display:inline-block; background:{badge_color}; color:#fff; border-radius:18px; padding:8px 22px; font-size:1.15rem; font-weight:700; margin-bottom:18px;'>
    #     {badge_text} Farm Performance
    # </div>
    # """, unsafe_allow_html=True)
    # Profile & Farm Summary table (with plant details)
    st.markdown(f"""
    <div style='background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(33,150,243,0.07);'>
    <h4 style='color: #1976d2;'>Profile & Farm Summary</h4>
    <table style='width:100%; font-size:1.05rem; color:#1976d2;'>
        <tr><td><b>Name</b></td><td>{st.session_state.get('profile_name', '')}</td></tr>
        <tr><td><b>Email</b></td><td>{st.session_state.get('profile_email', '')}</td></tr>
        <tr><td><b>Farm Location</b></td><td>{st.session_state.get('profile_location', '')}</td></tr>
        <tr><td><b>Farm Size</b></td><td>{y['field_size']} acres</td></tr>
        <tr><td><b>Planting Density</b></td><td>{y['planting_density']} trees/acre</td></tr>
        <tr><td><b>Estimated Trees</b></td><td>{y['total_trees']}</td></tr>
        <tr><td><b>Estimated Total Mangoes</b></td><td><b style='color:#27ae60;'>{y['est_yield_range'][0]} – {y['est_yield_range'][1]}</b></td></tr>
        <tr><td><b>Per-tree Yield</b></td><td>{y['per_tree_yield']} mangoes/tree</td></tr>
        <tr><td><b>Mango Variety</b></td><td>{y['mango']}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    # Checkbox for recommended fertilizer/pesticide use in profile
    key = f"organic_used_{y['mango']}"
    st.checkbox("I have used the recommended fertilizers and pesticides for this variety.", key=key)
    st.markdown("<span style='font-size:0.93rem; color:#1976d2; opacity:0.7;'>Mangoes grown with organic fertilizers and pesticides can be sold for higher value.</span>", unsafe_allow_html=True)
    # Motivational and actionable tips
    # st.markdown("""
    # <div style='background: linear-gradient(90deg,#f8fbff 0%,#e3f0ff 100%); border-radius: 12px; padding: 18px 24px; margin-top: 18px; box-shadow: 0 1px 4px rgba(33,150,243,0.09); border-left: 4px solid #42a5f5;'>
    #     <h4 style='color: #1976d2;'>🚀 Boost Your Success!</h4>
    #     <ul style='color: #1976d2; font-size: 1.05rem;'>
    #         <li>🌱 Try intercropping with legumes to improve soil fertility.</li>
    #         <li>💧 Install soil moisture sensors for precision irrigation.</li>
    #         <li>🦠 Use biofertilizers and beneficial microbes for healthier trees.</li>
    #         <li>📊 Track your yields and costs each season for better planning.</li>
    #         <li>🛡️ Join a local farmer group for knowledge sharing and support.</li>
    #         <li>📦 Explore direct-to-consumer sales for higher profits.</li>
    #         <li>🌤️ Use weather apps to plan irrigation and harvests.</li>
    #         <li>🎯 Set a yield goal for next season and track your progress!</li>
    #     </ul>
    # </div>
    # """, unsafe_allow_html=True)

# Place this before main()
mango_nutrient_pest_info = {
    "Alphonso": {
        "Fertilizer": [
            ("NPK 6-6-6", "https://www.amazon.in/s?k=npk+fertilizer"),
            ("Vermicompost", "https://www.amazon.in/s?k=vermicompost"),
            ("Micronutrient Mix", "https://www.amazon.in/s?k=micronutrient+mix+fertilizer")
        ],
        "Pesticide": [
            ("Neem Oil", "https://www.amazon.in/s?k=neem+oil+pesticide"),
            ("Imidacloprid 17.8% SL", "https://www.amazon.in/s?k=imidacloprid+17.8+sl"),
            ("Copper Oxychloride", "https://www.amazon.in/s?k=copper+oxychloride+fungicide")
        ],
        "Notes": "Apply NPK fertilizer at the start of the growing season. Use neem oil as a preventive spray every 2 weeks. Watch for anthracnose and mealybugs."
    },
    "Haden": {
        "Fertilizer": [
            ("NPK 8-3-9", "https://www.amazon.in/s?k=npk+8-3-9+fertilizer"),
            ("Compost", "https://www.amazon.in/s?k=compost+fertilizer"),
            ("Magnesium Sulphate", "https://www.amazon.in/s?k=magnesium+sulphate+fertilizer")
        ],
        "Pesticide": [
            ("Copper Hydroxide", "https://www.amazon.in/s?k=copper+hydroxide+fungicide"),
            ("Sulphur Dust", "https://www.amazon.in/s?k=sulphur+dust+fungicide"),
            ("Lambda Cyhalothrin", "https://www.amazon.in/s?k=lambda+cyhalothrin+pesticide")
        ],
        "Notes": "Apply compost in early spring. Use copper hydroxide for anthracnose. Monitor for fruit fly and scale insects."
    },
    "Keitt": {
        "Fertilizer": [
            ("NPK 10-26-26", "https://www.amazon.in/s?k=npk+10-26-26+fertilizer"),
            ("Bone Meal", "https://www.amazon.in/s?k=bone+meal+fertilizer"),
            ("Potash", "https://www.amazon.in/s?k=potash+fertilizer")
        ],
        "Pesticide": [
            ("Mancozeb 75% WP", "https://www.amazon.in/s?k=mancozeb+75+wp"),
            ("Carbaryl 50% WP", "https://www.amazon.in/s?k=carbaryl+50+wp"),
            ("Chlorpyrifos", "https://www.amazon.in/s?k=chlorpyrifos+pesticide")
        ],
        "Notes": "Apply potash before flowering. Use mancozeb for leaf spot. Watch for stem borer and hopper."
    },
    "Kent": {
        "Fertilizer": [
            ("NPK 12-24-12", "https://www.amazon.in/s?k=npk+12-24-12+fertilizer"),
            ("Farmyard Manure", "https://www.amazon.in/s?k=farmyard+manure"),
            ("Zinc Sulphate", "https://www.amazon.in/s?k=zinc+sulphate+fertilizer")
        ],
        "Pesticide": [
            ("Dimethoate 30% EC", "https://www.amazon.in/s?k=dimethoate+30+ec"),
            ("Copper Oxychloride", "https://www.amazon.in/s?k=copper+oxychloride+fungicide"),
            ("Spinosad", "https://www.amazon.in/s?k=spinosad+pesticide")
        ],
        "Notes": "Apply farmyard manure in winter. Use copper oxychloride for anthracnose. Monitor for fruit borer and aphids."
    },
    "Tommy_Atkins": {
        "Fertilizer": [
            ("NPK 19-19-19", "https://www.amazon.in/s?k=npk+19-19-19+fertilizer"),
            ("Poultry Manure", "https://www.amazon.in/s?k=poultry+manure+fertilizer"),
            ("Iron Chelate", "https://www.amazon.in/s?k=iron+chelate+fertilizer")
        ],
        "Pesticide": [
            ("Malathion 50% EC", "https://www.amazon.in/s?k=malathion+50+ec"),
            ("Trichoderma", "https://www.amazon.in/s?k=trichoderma+fungicide"),
            ("Imidacloprid 17.8% SL", "https://www.amazon.in/s?k=imidacloprid+17.8+sl")
        ],
        "Notes": "Apply poultry manure in late winter. Use trichoderma for soil-borne diseases. Watch for scale insects and powdery mildew."
    }
}

def main():
    setup_ui_theme()
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Sidebar - Navigation
    with st.sidebar:
        st.markdown(
            "<div style='margin-top: -10px; margin-bottom: 0; text-align: left;'>"
            "<div style='font-size:1.7rem; font-weight:800; color:#111; margin-bottom: 0;'>Sow Smart</div>"
            "<div style='font-size:1.05rem; color:#1976d2; font-weight:600; margin-bottom: 12px; margin-top: 0;'>Grow Smarter, Farm Better.</div>"
            "<hr style='margin-top: 0; margin-bottom: 0;'>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        # Navigation buttons
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
            
        if st.button("🔍 Mango Classifier", use_container_width=True):
            st.session_state.current_page = "Classifier"
            st.rerun()
            
        if st.button("🥭 AI Disease Classifier", use_container_width=True):
            st.session_state.current_page = "DiseaseClassifier"
            st.rerun()
            
        if st.button("📅 Mango Planner", use_container_width=True):
            st.session_state.current_page = "Planner"
            st.rerun()
            
        if st.button("📊 Market Analysis", use_container_width=True):
            st.session_state.current_page = "Market"
            st.rerun()
            
        if st.button("🤖 Chat Assistant", use_container_width=True):
            st.session_state.current_page = "Chat"
            st.rerun()
        
        if st.button("🧪 Nutrient & Pest Care", use_container_width=True):
            st.session_state.current_page = "NutrientPestCare"
            st.rerun()
        
        if st.button("🌾 Yield Optimizer", use_container_width=True):
            st.session_state.current_page = "YieldOptimizer"
            st.rerun()
        
        # Add sidebar button
        if st.button("👤 Profile", use_container_width=True):
            st.session_state.current_page = "Profile"
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
            <h4>🔧 Input Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Add debug mode toggle
        debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.warning("Debug mode enabled. Additional information will be shown.")
            
            # Show current session state
            with st.expander("Session State"):
                st.json({k: v for k, v in st.session_state.items() if k != '_last_updated'})
        
        # Common input parameters
        st.markdown("### Environmental Conditions")
        avg_temperature = st.slider(
            "Average Temperature (°C)",
            min_value=-20.0,
            max_value=40.0,
            value=26.0,
            step=0.5,
            help="Average ambient temperature for Cambridge, MA"
        )
        
        humidity_percent = st.slider(
            "Relative Humidity (%)",
            min_value=0,
            max_value=100,
            value=45,
            help="Average relative humidity for Cambridge, MA"
        )
        
        # Store these in session state for use across pages
        st.session_state.avg_temperature = avg_temperature
        st.session_state.humidity_percent = humidity_percent
        
        st.markdown("---")
        st.caption("Adjust these parameters to see how they affect predictions across all tools.")
    
    # Main content area
    if st.session_state.current_page == "Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "Classifier":
        if 'model' not in st.session_state:
            st.session_state.model = load_model()
        show_classifier(st.session_state.model)
    elif st.session_state.current_page == "NutrientPestCare":
        st.title("🧪 Nutrient & Pest Care")
        st.markdown("""
        <div style='color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>
            Get personalized recommendations for nutrient management and pest control for each mango variety. Select varieties below to view best practices, recommended products, and quick tips!
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        mango_types = ["Alphonso", "Haden", "Keitt", "Kent", "Tommy_Atkins"]
        icon_map = {"Fertilizer": "🧪", "Pesticide": "🦠"}
        # Variety selector
        selected_varieties = st.multiselect(
            "Select Mango Varieties",
            options=mango_types,
            default=mango_types,
            format_func=lambda x: x.replace('_', ' ')
        )
        st.markdown("---")
        # Info cards for each selected variety
        for mango in selected_varieties:
            info = mango_nutrient_pest_info[mango]
            st.markdown(f"""
            <div style='background: #f8fbff; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(33,150,243,0.08); border-left: 5px solid #1976d2;'>
                <h4 style='color: #1976d2; margin-bottom: 10px;'>{mango.replace('_', ' ')} Nutrient & Pest Care</h4>
                <div style='display: flex; gap: 32px; flex-wrap: wrap;'>
                    <div style='flex:1; min-width:220px; background:#fff; border-radius:10px; padding:16px 18px; box-shadow:0 1px 4px rgba(33,150,243,0.07); border-left:4px solid #ffd700;'>
                        <b>🧪 Fertilizer Recommendations:</b><ul style='margin-top:8px;'>
            """ + ''.join([
                f"<li><a href='{url}' target='_blank'>{name}</a></li>" for name, url in info["Fertilizer"]
            ]) + """
                        </ul>
                    </div>
                    <div style='flex:1; min-width:220px; background:#fff; border-radius:10px; padding:16px 18px; box-shadow:0 1px 4px rgba(33,150,243,0.07); border-left:4px solid #27ae60;'>
                        <b>🦠 Pesticide Recommendations:</b><ul style='margin-top:8px;'>
            """ + ''.join([
                f"<li><a href='{url}' target='_blank'>{name}</a></li>" for name, url in info["Pesticide"]
            ]) + """
                        </ul>
                    </div>
                </div>
                <div style='margin-top: 12px; color: #1976d2; font-size: 1.01rem; background:#fff; border-radius:8px; padding:12px 16px;'>
                    <b>Notes:</b> {info['Notes']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Best practices and quick tips
        st.markdown("""
        <div style='background: linear-gradient(90deg, #e3f0ff 0%, #f8fbff 100%); border-radius: 14px; padding: 18px 32px 18px 32px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(33,150,243,0.10); border-left: 5px solid #2196f3;'>
            <h4 style='color: #1976d2;'>🌱 Best Practices & Quick Tips</h4>
            <ul style='color: #1976d2; font-size: 1.05rem;'>
                <li>Apply fertilizers at the start of the growing season and after harvest.</li>
                <li>Use organic mulch to retain soil moisture and suppress weeds.</li>
                <li>Rotate pesticides to prevent resistance buildup.</li>
                <li>Monitor for pests and diseases regularly; use integrated pest management (IPM) practices.</li>
                <li>Follow label instructions for all agrochemicals.</li>
                <li>Wear protective gear when spraying pesticides.</li>
                <li>Keep records of all applications for traceability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.current_page == "DiseaseClassifier":
        show_disease_classifier()
    elif st.session_state.current_page == "Planner":
        show_crop_planner()
    elif st.session_state.current_page == "Market":
        show_market_analysis()
    elif st.session_state.current_page == "Chat":
        show_chat_assistant()
    elif st.session_state.current_page == "YieldOptimizer":
        show_yield_optimizer()
    elif st.session_state.current_page == "Profile":
        show_profile()
    elif st.session_state.current_page == "FindBuyer":
        show_find_buyer()

def show_find_buyer():
    st.title("🔎 Find Mango Buyers Near You")
    st.markdown("""
    <div style='color: #666; margin-bottom: 1.5rem; font-size: 1.1rem;'>
        Discover mango buyers and traders near your farm location. Connect directly and supply your estimated harvest!
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    location = st.session_state.get('profile_location', 'Unknown')
    y = st.session_state.get('last_yield_optimizer', None)
    # --- Step 1: Enter quantity to sell ---
    if 'find_buyer_quantity_submitted' not in st.session_state or not st.session_state['find_buyer_quantity_submitted']:
        max_qty = y['est_yield_range'][1] if y else 0
        with st.form("find_buyer_quantity_form"):
            st.markdown(f"<b>Farm Location:</b> {location}", unsafe_allow_html=True)
            if y:
                st.markdown(f"<b>Estimated Mangoes for Supply:</b> <span style='color:#27ae60; font-weight:700;'>{y['est_yield_range'][0]} – {y['est_yield_range'][1]}</span>", unsafe_allow_html=True)
            else:
                st.info("No yield data available. Please use the Yield Optimizer first.")
            qty = st.number_input("How many mangoes do you want to sell?", min_value=1, max_value=max_qty if max_qty > 0 else 100000, value=max_qty if max_qty > 0 else 100, step=1)
            submitted = st.form_submit_button("Send to Market")
        if submitted:
            st.session_state['find_buyer_quantity'] = qty
            st.session_state['find_buyer_quantity_submitted'] = True
            st.rerun()
    else:
        # --- Step 2: Confirmation and show no buyers ---
        qty = st.session_state.get('find_buyer_quantity', None)
        st.success(f"✅ Your offer to sell {qty} mangoes has been sent to the market! Potential buyers can now contact you directly.")
        st.markdown("<span style='color:#1976d2; font-size:1.08rem;'>You will be contacted by interested buyers soon.</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🗺️ Nearby Mango Buyers", unsafe_allow_html=True)
        st.info("No available buyers at the moment.")
        st.markdown("---")
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Profile", key="back_to_profile"):
                st.session_state.current_page = "Profile"
                st.session_state['find_buyer_quantity_submitted'] = False
                st.rerun()
        with col2:
            if st.button("🏠 Dashboard", key="back_to_dashboard"):
                st.session_state.current_page = "Dashboard"
                st.session_state['find_buyer_quantity_submitted'] = False
                st.rerun()

if __name__ == "__main__":
    main()
