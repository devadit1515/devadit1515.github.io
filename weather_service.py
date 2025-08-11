import requests
import streamlit as st
import json
from datetime import datetime, timedelta
from config import WEATHERSTACK_API_KEY, DEFAULT_LOCATION

# Enable debug logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
WEATHERSTACK_BASE_URL = "http://api.weatherstack.com/current"
CACHE_TTL_MINUTES = 30  # Cache weather data for 30 minutes

@st.cache_data(ttl=CACHE_TTL_MINUTES * 60)  # Convert to seconds
def get_weather_data(location=DEFAULT_LOCATION):
    """
    Fetch current weather data from Weatherstack API
    """
    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": location,
        "units": "m"  # Metric units
    }
    
    logger.info(f"Fetching weather data for location: {location}")
    logger.debug(f"API URL: {WEATHERSTACK_BASE_URL}")
    
    try:
        response = requests.get(WEATHERSTACK_BASE_URL, params=params, timeout=10)
        logger.debug(f"Response status: {response.status_code}")
        
        # Log the raw response for debugging
        logger.debug(f"Raw response: {response.text[:500]}...")  # Log first 500 chars
        
        response.raise_for_status()
        data = response.json()
        
        # Log the full response in debug mode
        logger.debug(f"Parsed response: {json.dumps(data, indent=2)}")
        
        if 'error' in data:
            error_info = data.get('error', {})
            error_msg = f"Weather API Error {error_info.get('code', '')}: {error_info.get('info', 'Unknown error')}"
            logger.error(f"API Error: {error_msg}")
            st.error(error_msg)
            return None
            
        if 'current' not in data:
            logger.error("Unexpected API response format: 'current' key missing")
            logger.error(f"Response keys: {data.keys()}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        st.error(f"Error fetching weather data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response: {str(e)}")
        st.error("Error parsing weather data. Please try again later.")
        return None

def format_weather_data(weather_data):
    """
    Format the raw weather data into a more usable structure
    """
    if not weather_data:
        logger.warning("No weather data provided to format_weather_data")
        return None
        
    if 'current' not in weather_data:
        logger.error(f"'current' key missing in weather_data. Available keys: {weather_data.keys()}")
        return None
        
    current = weather_data['current']
    location = weather_data.get('location', {})
    
    # Safely get weather descriptions and icons
    weather_descriptions = current.get('weather_descriptions', ['N/A'])
    weather_icons = current.get('weather_icons', [''])
    
    # Format wind direction from degrees to cardinal direction
    wind_degree = current.get('wind_degree', 0)
    wind_dir = current.get('wind_dir', '')
    if not wind_dir and wind_degree is not None:
        # Convert degrees to cardinal direction if not provided
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        wind_dir = directions[round(wind_degree / 22.5) % 16] if wind_degree is not None else ''
    
    # Format the data with fallbacks
    formatted_data = {
        'location': f"{location.get('name', 'Unknown')}, {location.get('country', 'Unknown')}",
        'temperature': current.get('temperature', 'N/A'),
        'feelslike': current.get('feelslike', current.get('temperature', 'N/A')),  # Fallback to temperature if feelslike not available
        'weather_desc': weather_descriptions[0] if weather_descriptions else 'N/A',
        'weather_icon': weather_icons[0] if weather_icons else None,
        'humidity': current.get('humidity', 'N/A'),
        'wind_speed': current.get('wind_speed', 0),
        'wind_dir': wind_dir,
        'wind_degree': wind_degree,
        'pressure': current.get('pressure', 'N/A'),
        'precip': current.get('precip', 0),
        'uv_index': current.get('uv_index', 'N/A'),
        'visibility': current.get('visibility', 'N/A'),
        'observation_time': current.get('observation_time', 'N/A'),
        'local_time': location.get('localtime', 'N/A'),
        'air_quality': current.get('air_quality', {})
    }
    
    # Log the formatted data for debugging
    logger.debug(f"Formatted weather data: {json.dumps(formatted_data, indent=2, default=str)}")
    
    return formatted_data
