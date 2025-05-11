import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from sklearn.ensemble import RandomForestRegressor # For dummy model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import datetime
import os
import json
import geopandas as gpd
import xgboost as xgb
# --- Configuration & Constants ---
DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "app_files/xgboost_model.pkl")
SHAP_VALUES_FILE = os.path.join(DATA_DIR, "app_files/shap_values.csv")
GEOJSON_FILE = os.path.join(DATA_DIR, "app_files/ward/London_Ward.shp")

LONDON_CENTER = [51.5074, -0.1278] # Latitude, Longitude
DEFAULT_ZOOM = 10
BOROUGH_ZOOM = 12
TOTAL_PATROL_HOURS_PER_BOROUGH_MONTH = 200 # Global constraint for patrol allocation

# --- Dummy Data Generation (if real files are not found) ---
def create_dummy_geojson():
    """Creates a dummy GeoJSON object for London wards."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"GSS_CODE": "W01", "ward_name": "Westminster Ward A", "borough": "Westminster"},
                "geometry": {"type": "Polygon", "coordinates": [[[-0.14, 51.51], [-0.14, 51.52], [-0.12, 51.52], [-0.12, 51.51], [-0.14, 51.51]]]}
            },
            {
                "type": "Feature",
                "properties": {"GSS_CODE": "W02", "ward_name": "Westminster Ward B", "borough": "Westminster"},
                "geometry": {"type": "Polygon", "coordinates": [[[-0.12, 51.51], [-0.12, 51.52], [-0.10, 51.52], [-0.10, 51.51], [-0.12, 51.51]]]}
            },
            {
                "type": "Feature",
                "properties": {"GSS_CODE": "K01", "ward_name": "Kensington Ward A", "borough": "Kensington"},
                "geometry": {"type": "Polygon", "coordinates": [[[-0.20, 51.50], [-0.20, 51.51], [-0.18, 51.51], [-0.18, 51.50], [-0.20, 51.50]]]}
            },
            {
                "type": "Feature",
                "properties": {"GSS_CODE": "I01", "ward_name": "Islington Ward A", "borough": "Islington"},
                "geometry": {"type": "Polygon", "coordinates": [[[-0.11, 51.53], [-0.11, 51.54], [-0.09, 51.54], [-0.09, 51.53], [-0.11, 51.53]]]}
            }
        ]
    }

def create_dummy_shap_values(ward_ids, months):
    """Creates a dummy SHAP values DataFrame."""
    data = []
    features = [("Unemployment Rate", 0.1, 0.5), ("Proximity to Transit", 0.05, 0.3), ("Housing Density", 0.2, 0.6), ("Previous Month Burglaries", 1, 10), ("Avg. Income", 20000, 50000)]
    for ward in ward_ids:
        for month in months:
            # Simulate top 3 features
            top_features = np.random.choice(len(features), size=3, replace=False)
            row = {"ward_id": ward, "month": month}
            for i, idx in enumerate(top_features):
                feature_name, min_val, max_val = features[idx]
                row[f"feature{i+1}_name"] = feature_name
                if isinstance(min_val, float):
                    row[f"feature{i+1}_value"] = f"{np.random.uniform(min_val, max_val):.2f}"
                else:
                    row[f"feature{i+1}_value"] = f"{np.random.randint(min_val, max_val)}"
            data.append(row)
    return pd.DataFrame(data)

def create_dummy_model_instance():
    """Creates and returns a new dummy XGBoost model instance (in-memory)."""
    # Generate dummy data with the expected features
    feature_names = [
        'Population', 'area_km2', 'claimant_rate', 'IncScore', 'month_nr',
        'burglary_lag_1', 'burglary_lag_3', 'burglary_lag_12',
        'burglary_rolling_mean_3', 'burglary_volatility_3',
        'burglary_rolling_mean_6', 'burglary_rolling_std_6',
        'burglary_rolling_mean_12', 'burglary_volatility_12',
        'burglary_rolling_max_3', 'burglary_rolling_min_3',
        'burglary_rolling_max_6', 'burglary_rolling_min_6',
        'burglary_rolling_max_12', 'burglary_rolling_min_12',
        'burglary_trend_3_12', 'burglary_trend_6_12'
    ]
    X, y = make_regression(n_samples=100, n_features=len(feature_names), random_state=42)
    # Create XGBoost model
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X, y)
    # Manually set feature_names (XGBoost doesn't automatically store them)
    model.feature_names = feature_names
    print("In-memory dummy XGBoost model instance created with expected feature names.")
    return model

# --- Data Loading Functions ---
@st.cache_data
def load_model():
    """Loads the trained forecasting model or creates a dummy if not found/corrupted."""
    try:
        if os.path.exists(MODEL_FILE):
            # Attempt to load existing model
            model = joblib.load(MODEL_FILE)
            MODEL_FEATURES = model.feature_names
            # print(f"Model loaded from {MODEL_FILE}") # Optional: for console logging
            return model
        else:
            # Model file does not exist, create a new dummy model
            st.warning(f"Model file not found at {MODEL_FILE}. Creating and saving a new dummy model.")
            os.makedirs(DATA_DIR, exist_ok=True)
            dummy_model = create_dummy_model_instance()
            joblib.dump(dummy_model, MODEL_FILE)
            # print(f"New dummy model created and saved to {MODEL_FILE}") # Optional: for console logging
            return dummy_model
    except Exception as e:
        # Error loading existing file (e.g., corrupted) or other issue
        st.error(f"Error loading model from {MODEL_FILE}: {e}. Creating and saving a new dummy model as fallback.")
        os.makedirs(DATA_DIR, exist_ok=True)
        dummy_model = create_dummy_model_instance()
        try:
            joblib.dump(dummy_model, MODEL_FILE) # Try to save the new dummy model, overwriting if necessary
            # print(f"New dummy model created and saved to {MODEL_FILE} after load failure.") # Optional: for console logging
        except Exception as save_e:
            st.error(f"Could not save the new dummy model to {MODEL_FILE}: {save_e}")
        return dummy_model

@st.cache_data
def load_sarima_model():
    """Placeholder for loading a time-series model (e.g., SARIMA)."""
    st.info("SARIMA model integration is a placeholder. This function is not yet implemented.")
    # In the future, load your SARIMA model here.
    # Example:
    # sarima_model = joblib.load(os.path.join(DATA_DIR, "sarima_model.joblib"))
    # return sarima_model
    return None

# --- Fixed loader ---
@st.cache_data
def load_geojson_data():
    if not os.path.exists(GEOJSON_FILE):
        st.warning(f"Map file not found at {GEOJSON_FILE}. Using dummy boundaries.")
        return create_dummy_geojson()

    ext = os.path.splitext(GEOJSON_FILE)[1].lower()
    try:
        if ext == ".shp":
            gdf = gpd.read_file(GEOJSON_FILE).to_crs(epsg=4326)
            return json.loads(gdf.to_json())
        elif ext in (".geojson", ".json"):
            with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.error(f"Unsupported map format {ext}. Using dummy boundaries.")
            return create_dummy_geojson()
    except Exception as e:
        st.error(f"Error loading map file: {e}. Using dummy boundaries.")
        return create_dummy_geojson()

@st.cache_data
def load_shap_values_data(_ward_ids, _months): # Add arguments to make cache sensitive to changes
    """Loads SHAP values from shap_values.csv."""
    if not os.path.exists(SHAP_VALUES_FILE):
        #st.warning(f"SHAP values file not found at {SHAP_VALUES_FILE}. Using dummy SHAP data.")
        geojson_data = load_geojson_data()
        ward_ids = [feature['properties']['GSS_CODE'] for feature in geojson_data['features']]
        # Generate for a range of months to support spike calculation
        months_for_dummy_shap = []
        start_date = datetime.date(2024, 1, 1)
        for i in range(24): # 2 years of dummy data
             months_for_dummy_shap.append((start_date + datetime.timedelta(days=i*30)).strftime('%Y-%m'))

        return create_dummy_shap_values(ward_ids, months_for_dummy_shap)
    try:
        return pd.read_csv(SHAP_VALUES_FILE)
    except Exception as e:
        #st.error(f"Error loading SHAP values: {e}. Using dummy SHAP data.")
        geojson_data = load_geojson_data()
        ward_ids = [feature['properties']['GSS_CODE'] for feature in geojson_data['features']]
        months_for_dummy_shap = []
        start_date = datetime.date(2024, 1, 1)
        for i in range(24): # 2 years of dummy data
             months_for_dummy_shap.append((start_date + datetime.timedelta(days=i*30)).strftime('%Y-%m'))
        return create_dummy_shap_values(ward_ids, months_for_dummy_shap)


# --- Prediction and Feature Engineering ---
def engineer_features_for_ward_month(ward_id, target_month_str):
    """
    Generates features for a given ward and month, matching the model's expected features.
    In a real scenario, fetch data from a database or files.
    For this PoC, we generate dummy values for the required features.
    """
    # Ensure target_month_str is in 'YYYY-MM' format
    year, month = map(int, target_month_str.split('-'))

    # Set random seed for reproducibility
    np.random.seed(hash(ward_id + target_month_str) % (2**32))

    # Define the features expected by the model (from the error message)
    features = {
        'Population': np.random.randint(5000, 50000),
        'population_density': np.random.randint(500, 5000),# Population density (500-5000 per km2)
        'poi_count': np.random.randint(0, 300), # Point of interest count (0-300)
        'area_km2': np.random.uniform(0.5, 10.0),      # Area in square kilometers
        'claimant_rate': np.random.uniform(0.01, 0.1), # Unemployment/claimant rate (1-10%)
        'IncScore': np.random.uniform(0, 100),         # Income score (arbitrary scale)
        'month_nr': month,                             # Month number (1-12)
        'burglary_lag_1': np.random.randint(0, 10),    # Burglaries in previous month
        'burglary_lag_3': np.random.randint(0, 10),    # Burglaries 3 months ago
        'burglary_lag_12': np.random.randint(0, 10),   # Burglaries 12 months ago
        'burglary_rolling_mean_3': np.random.uniform(0, 10),  # 3-month rolling mean
        'burglary_volatility_3': np.random.uniform(0, 5),     # 3-month volatility
        'burglary_rolling_mean_6': np.random.uniform(0, 10),  # 6-month rolling mean
        'burglary_rolling_std_6': np.random.uniform(0, 5),    # 6-month rolling std
        'burglary_rolling_mean_12': np.random.uniform(0, 10), # 12-month rolling mean
        'burglary_volatility_12': np.random.uniform(0, 5),    # 12-month volatility
        'burglary_rolling_max_3': np.random.randint(5, 15),   # 3-month rolling max
        'burglary_rolling_min_3': np.random.randint(0, 5),    # 3-month rolling min
        'burglary_rolling_max_6': np.random.randint(5, 15),   # 6-month rolling max
        'burglary_rolling_min_6': np.random.randint(0, 5),    # 6-month rolling min
        'burglary_rolling_max_12': np.random.randint(5, 15),  # 12-month rolling max
        'burglary_rolling_min_12': np.random.randint(0, 5),   # 12-month rolling min
        'burglary_trend_3_12': np.random.uniform(-1, 1),      # Trend between 3 and 12 months
        'burglary_trend_6_12': np.random.uniform(-1, 1),      # Trend between 6 and 12 months
    }

    # Create a DataFrame with one row
    feature_vector = pd.DataFrame([features])
    return feature_vector

# def predict_risk(model, ward_id, target_month_str):
#     X = engineer_features_for_ward_month(ward_id, target_month_str)

#     # try scikit-style API first
#     try:
#         pred = model.predict(X)
#     except TypeError:
#         # if that fails, fallback to raw Booster API
#         dmat = xgb.DMatrix(X)
#         pred = model.predict(dmat)
#     # both APIs return arrays
#     return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)

def predict_risk(model, ward_id, target_month_str):
    features_df = engineer_features_for_ward_month(ward_id, target_month_str)
    MODEL_FEATURES = model.feature_names
    print(f"Model expected features: {MODEL_FEATURES}")  # Debug statement
    print(f"Features provided: {features_df.columns.tolist()}")  # Debug statement
    # Select only the model's expected features
    X = features_df[MODEL_FEATURES]
    # Wrap in DMatrix
    dm = xgb.DMatrix(X)
    pred = model.predict(dm)[0]
    return pred


def get_all_ward_predictions(model, ward_ids, target_month_str):
    """Generates predictions for all wards for a given month."""
    predictions = {}
    for ward_id in ward_ids:
        predictions[ward_id] = predict_risk(model, ward_id, target_month_str)
    return predictions

# --- Core Logic Functions ---
def allocate_patrol_hours(ward_risk_scores_df, total_hours_constraint):
    """
    Allocates patrol hours per ward based on predicted risk.
    Currently uses proportional allocation.
    Structure allows for easy replacement of allocation logic.

    Args:
        ward_risk_scores_df (pd.DataFrame): DataFrame with 'ward_id' and 'risk_score'.
        total_hours_constraint (float): Total available patrol hours.

    Returns:
        pd.DataFrame: DataFrame with 'ward_id', 'risk_score', 'allocated_hours'.
    """
    if ward_risk_scores_df.empty or ward_risk_scores_df['risk_score'].sum() == 0:
        # Assign equal hours if no risk or empty data, or zero total risk
        if not ward_risk_scores_df.empty:
            equal_hours = total_hours_constraint / len(ward_risk_scores_df) if len(ward_risk_scores_df) > 0 else 0
            ward_risk_scores_df['allocated_hours'] = equal_hours
            return ward_risk_scores_df
        else:
            return pd.DataFrame(columns=['ward_id', 'risk_score', 'allocated_hours'])


    # --- Proportional Allocation Logic ---
    # This is where you can replace the allocation rule.
    # E.g., tiered allocation, fairness-aware allocation, optimization-based.
    total_risk = ward_risk_scores_df['risk_score'].sum()
    ward_risk_scores_df['allocated_hours'] = (ward_risk_scores_df['risk_score'] / total_risk) * total_hours_constraint
    # --- End of Proportional Allocation Logic ---

    ward_risk_scores_df['allocated_hours'] = ward_risk_scores_df['allocated_hours'].round(1)
    return ward_risk_scores_df

def compute_risk_spikes(current_risk_df, previous_risk_df):
    """
    Computes month-over-month change in predicted risk and identifies top 5 spikes.

    Args:
        current_risk_df (pd.DataFrame): 'ward_id', 'risk_score' for the current month.
        previous_risk_df (pd.DataFrame): 'ward_id', 'risk_score' for the previous month.

    Returns:
        pd.DataFrame: Top 5 wards with largest risk increase, including 'Ward Name',
                      'Last Month Risk', 'This Month Risk', 'Delta'.
    """
    if previous_risk_df.empty or current_risk_df.empty:
        return pd.DataFrame(columns=['Ward Name', 'Last Month Risk', 'This Month Risk', 'Delta'])

    merged_df = pd.merge(current_risk_df, previous_risk_df, on='ward_id', suffixes=('_current', '_previous'))
    merged_df['Delta'] = merged_df['risk_score_current'] - merged_df['risk_score_previous']
    merged_df = merged_df.sort_values(by='Delta', ascending=False).head(5)

    # Add ward names (assuming geojson_data is available and has 'ward_name')
    geojson_data = load_geojson_data()
    ward_id_to_name = {feat['properties']['GSS_CODE']: feat['properties'].get('GSS_NAME', feat['properties']['GSS_CODE'])
                       for feat in geojson_data['features']}
    merged_df['Ward Name'] = merged_df['ward_id'].map(ward_id_to_name)

    return merged_df[['Ward Name', 'risk_score_previous', 'risk_score_current', 'Delta']] \
        .rename(columns={'risk_score_previous': 'Last Month Risk', 'risk_score_current': 'This Month Risk'})


# --- UI Components ---
def display_map(geojson_data, risk_data_df, shap_data_df, selected_month_str, center, zoom, selected_borough_wards=None):
    """
    Displays an interactive Folium map with a choropleth layer for risk scores
    and tooltips for SHAP values.

    Args:
        geojson_data (dict): GeoJSON data for ward boundaries.
        risk_data_df (pd.DataFrame): DataFrame with 'ward_id' and 'risk_score'.
        shap_data_df (pd.DataFrame): DataFrame with SHAP values by 'ward_id' and 'month'.
        selected_month_str (str): The selected month ('YYYY-MM').
        center (list): Initial center of the map [lat, lon].
        zoom (int): Initial zoom level of the map.
        selected_borough_wards (list, optional): List of ward_ids in the selected borough.
    """
    if risk_data_df.empty:
        st.warning("No risk data available to display on the map for the selected criteria.")
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
        st_folium(m, width=700, height=500)
        return

    # Filter SHAP data for the selected month
    shap_month_df = shap_data_df[shap_data_df['month'] == selected_month_str]

    # Create a Folium map
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    # Prepare data for choropleth by merging risk scores into GeoJSON properties
    # This assumes GeoJSON features have a 'properties' dict and a 'ward_id' key within it.
    # And risk_data_df has 'ward_id' and 'risk_score'
    risk_dict = risk_data_df.set_index('ward_id')['risk_score']

    # Filter GeoJSON features if a borough is selected
    display_geojson = geojson_data.copy()
    if selected_borough_wards:
        display_geojson['features'] = [f for f in geojson_data['features']
                                       if f['properties']['GSS_CODE'] in selected_borough_wards]
        if not display_geojson['features']:
            st.warning(f"No GeoJSON features found for the selected borough's wards.")
            # Still show the base map centered
            st_folium(m, width=700, height=500, returned_objects=[])
            return


    # Add choropleth layer
    choropleth = folium.Choropleth(
        geo_data=display_geojson,
        name='Burglary Risk',
        data=risk_data_df, # Pass the dataframe
        columns=['ward_id', 'risk_score'], # Columns to use from the dataframe
        key_on='feature.properties.GSS_CODE', # Path to the id in GeoJSON
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Predicted Burglary Risk Score ({selected_month_str})',
        highlight=True
    ).add_to(m)

    # Add tooltips with SHAP values
    # Tooltip content will be generated based on shap_month_df
    # Ensure ward_id in SHAP data matches ward_id in GeoJSON properties
    for feature in choropleth.geojson.data['features']:
        ward_id = feature['properties']['GSS_CODE']
        ward_name = feature['properties'].get('NAME', ward_id) # Name IS GOOD NOT GSS_NAME
        risk_score = risk_dict.get(ward_id, 'N/A')
        if isinstance(risk_score, (float, int)):
            risk_score = f"{risk_score:.2f}"


        tooltip_html = f"<b>Ward:</b> {ward_name} ({ward_id})<br>"
        tooltip_html += f"<b>Predicted Risk:</b> {risk_score}<br><hr>"
        tooltip_html += "<b>Top Risk Drivers:</b><br>"

        shap_info = shap_month_df[shap_month_df['ward_id'] == ward_id]
        if not shap_info.empty:
            shap_row = shap_info.iloc[0]
            for i in range(1, 4): # Assuming feature1_name, feature1_value, etc.
                feature_name = shap_row.get(f'feature{i}_name', None)
                feature_val = shap_row.get(f'feature{i}_value', None)
                if feature_name and feature_val is not None:
                    tooltip_html += f"&bull; {feature_name}: {feature_val}<br>"
        else:
            tooltip_html += "No feature importance data available for this ward/month."

        folium.GeoJsonTooltip(fields=['NAME'], # Basic field from properties for initial hover
                              aliases=['Ward:'], # Label for that field
                              sticky=False,
                              labels=True,
                              localize=True,
                              style="""
                                background-color: #F0EFEF;
                                border: 2px solid black;
                                border-radius: 3px;
                                box-shadow: 3px;
                            """
                             ).add_to(choropleth.geojson) # Add to the choropleth's geojson layer

        # For more complex popups on click, use folium.Popup
        # This current implementation uses tooltips which appear on hover.
        # For click-based popups with SHAP:
        popup = folium.Popup(tooltip_html, max_width=300)
        popup.add_to(choropleth.geojson) # This adds popup on click

    # Fit map to bounds of the displayed GeoJSON
    if display_geojson['features']:
        # Calculate bounds of the filtered geojson
        all_coords = []
        for f in display_geojson['features']:
            coords = f['geometry']['coordinates']
            # Handle Polygon and MultiPolygon
            if f['geometry']['type'] == 'Polygon':
                all_coords.extend(coords[0]) # Exterior ring
            elif f['geometry']['type'] == 'MultiPolygon':
                for poly in coords:
                    all_coords.extend(poly[0]) # Exterior ring of each polygon part

        if all_coords: # Check if any coordinates were actually added
            # Folium's fit_bounds expects [[south, west], [north, east]]
            min_lon = min(c[0] for c in all_coords)
            min_lat = min(c[1] for c in all_coords)
            max_lon = max(c[0] for c in all_coords)
            max_lat = max(c[1] for c in all_coords)
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])


    folium.LayerControl().add_to(m)
    return st_folium(m, width=725, height=500, returned_objects=['last_active_drawing'])


# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Burglary Risk & Patrol Allocation")

    # --- Load Data ---
    model = load_model()
    # sarima_model = load_sarima_model() # Placeholder
    geojson_data = load_geojson_data()
    ward_ids_from_geojson = [feature['properties']['GSS_CODE'] for feature in geojson_data['features']]
    all_boroughs = sorted(list(set(feat['properties'].get('borough', 'Unknown Borough') for feat in geojson_data['features'])))

    # Create dummy SHAP data based on geojson wards and a date range
    # The arguments help @st.cache_data to rerun if ward_ids or months change
    # For a real app, you might load SHAP for ALL possible months or filter later.
    # Here, we generate dummy data that covers a potential range.
    shap_values_df = load_shap_values_data(tuple(ward_ids_from_geojson), None) # Pass tuple for hashability

    st.sidebar.title("Met Police PoC Dashboard")
    st.sidebar.markdown("Configure forecast parameters:")

    # --- User Inputs (Sidebar) ---
    # 1. Date Selector
    current_year = datetime.date.today().year
    # Create a list of months for selection (e.g., next 12 months from now)
    available_months = [(datetime.date(current_year -1 , 1, 1) + datetime.timedelta(days=30*i)).strftime("%Y-%m") for i in range(36)] # Past year and next 2 years
    default_month_index = 12 # Default to current month or a bit in future for forecasting
    if f"{datetime.date.today():%Y-%m}" in available_months:
        default_month_index = available_months.index(f"{datetime.date.today():%Y-%m}")


    selected_month_str = st.sidebar.selectbox(
        "Select Forecast Month:",
        options=available_months,
        index=default_month_index,
        help="Choose the month for which to forecast burglary risk."
    )
    selected_year, selected_month_num = map(int, selected_month_str.split('-'))

    # 2. Borough/Ward Selector
    selected_borough = st.sidebar.selectbox(
        "Select Borough (or All London):",
        options=["All London"] + all_boroughs,
        index=0,
        help="Zoom to a specific borough or view all of London."
    )

    # --- Main Area ---
    st.title("Residential Burglary Risk Forecast & Patrol Allocation")
    st.markdown(f"Displaying predictions and allocations for **{selected_month_str}**.")
    if selected_borough != "All London":
        st.markdown(f"Focused on **{selected_borough}**.")

    # --- Filter data based on selected borough ---
    map_center = LONDON_CENTER
    map_zoom = DEFAULT_ZOOM
    wards_in_scope_ids = ward_ids_from_geojson # Initially all wards

    if selected_borough != "All London":
        wards_in_scope_ids = [
            feat['properties']['ward_id'] for feat in geojson_data['features']
            if feat['properties'].get('borough') == selected_borough
        ]
        if not wards_in_scope_ids:
            st.warning(f"No wards found for borough: {selected_borough}. Displaying all of London.")
            wards_in_scope_ids = ward_ids_from_geojson # Fallback
        else:
            # Attempt to find a center for the borough (e.g., centroid of first ward)
            # A more robust way would be to precompute borough centroids
            first_ward_geom = next((f['geometry'] for f in geojson_data['features'] if f['properties']['ward_id'] == wards_in_scope_ids[0]), None)
            if first_ward_geom and first_ward_geom['type'] == 'Polygon':
                coords = first_ward_geom['coordinates'][0]
                map_center = [np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])] # avg_lat, avg_lon
            map_zoom = BOROUGH_ZOOM

    if not wards_in_scope_ids:
        st.error("No ward data available. Please check your GeoJSON file.")
        return

    # --- Generate Predictions for selected month and previous month ---
    # Current month predictions (for wards in scope)
    current_month_predictions = get_all_ward_predictions(model, wards_in_scope_ids, selected_month_str)
    current_risk_df = pd.DataFrame(list(current_month_predictions.items()), columns=['ward_id', 'risk_score'])

    # Previous month predictions (for all wards initially, for spike calculation flexibility)
    prev_month_date = datetime.date(selected_year, selected_month_num, 1) - datetime.timedelta(days=1)
    prev_month_str = prev_month_date.strftime("%Y-%m")
    # For spikes, we need previous month data for *all* wards, then filter later if borough selected
    previous_month_predictions_all_wards = get_all_ward_predictions(model, ward_ids_from_geojson, prev_month_str)
    previous_risk_df_all_wards = pd.DataFrame(list(previous_month_predictions_all_wards.items()), columns=['ward_id', 'risk_score'])


    # --- Display Components ---
    col1, col2 = st.columns([3, 1]) # Map takes more space

    with col1:
        st.subheader("Predicted Burglary Risk Hotspots")
        # Pass only wards_in_scope_ids to display_map if a borough is selected
        # The risk_data_df should also be filtered if we only want to show data for that borough on the map coloring
        # But the choropleth itself can handle filtering by key_on from a larger risk_data_df
        # For simplicity, let's pass all current_risk_df and let choropleth handle it,
        # but selected_borough_wards will filter the geojson features drawn.
        display_map(geojson_data, current_risk_df, shap_values_df, selected_month_str, map_center, map_zoom,
                    selected_borough_wards=wards_in_scope_ids if selected_borough != "All London" else None)

    with col2:
        st.subheader("Top 5 Risk Spike Wards")
        st.markdown(f"Comparing {selected_month_str} with {prev_month_str}")
        # Risk spikes should be calculated based on the selected scope (borough or all London)
        current_risk_for_spike_calc = current_risk_df # Already filtered if borough selected
        previous_risk_for_spike_calc = previous_risk_df_all_wards[previous_risk_df_all_wards['ward_id'].isin(wards_in_scope_ids)]

        top_spikes_df = compute_risk_spikes(current_risk_for_spike_calc, previous_risk_for_spike_calc)
        if not top_spikes_df.empty:
            st.dataframe(top_spikes_df.style.format({
                "Last Month Risk": "{:.2f}",
                "This Month Risk": "{:.2f}",
                "Delta": "{:.2f}"
            }), height=210) # Adjust height as needed
        else:
            st.info("Not enough data to compute risk spikes for the selected scope.")

    st.divider()

    st.subheader("Patrol Allocation Simulation")
    if not current_risk_df.empty:
        # Patrol allocation should be based on wards in the selected scope
        patrol_allocation_df = allocate_patrol_hours(current_risk_df.copy(), TOTAL_PATROL_HOURS_PER_BOROUGH_MONTH)

        # Add ward names for display
        ward_id_to_name = {feat['properties']['GSS_CODE']: feat['properties'].get('NAME', feat['properties']['GSS_CODE'])
                           for feat in geojson_data['features']}
        patrol_allocation_df['ward_name'] = patrol_allocation_df['ward_id'].map(ward_id_to_name)
        patrol_allocation_df = patrol_allocation_df[['ward_name', 'risk_score', 'allocated_hours']]


        st.markdown(f"Recommended patrol hours based on proportional risk (Total: {TOTAL_PATROL_HOURS_PER_BOROUGH_MONTH} hours for selected scope).")
        # Display as a table or bar chart
        # st.table(patrol_allocation_df.style.format({"risk_score": "{:.2f}", "allocated_hours": "{:.1f}"}))

        # Bar chart might be more visual
        if not patrol_allocation_df.empty:
            chart_data = patrol_allocation_df.set_index('ward_name')[['allocated_hours']]
            st.bar_chart(chart_data, height=400)
            with st.expander("View Allocation Data Table"):
                st.dataframe(patrol_allocation_df.style.format({"risk_score": "{:.2f}", "allocated_hours": "{:.1f}"}))
        else:
            st.info("No wards to allocate patrol hours for in the current selection.")

    else:
        st.info("No risk predictions available for the selected month and scope to allocate patrol hours.")


    st.divider()
    st.sidebar.info("This is a Proof-of-Concept application.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Future Integrations:**")
    st.sidebar.markdown("- SARIMA time-series model")
    st.sidebar.markdown("- Advanced patrol allocation rules (e.g., tiered, fairness-aware)")

if __name__ == "__main__":
    # The load_model() function now handles creation and saving if the model file is missing or corrupt.
    # So, explicit creation here is no longer needed.
    # if not os.path.exists(MODEL_FILE):
    #     print("Attempting to create dummy model as it's missing...")
    #     create_dummy_model() # This line is removed

    if not os.path.exists(GEOJSON_FILE):
         print(f"Warning: {GEOJSON_FILE} not found. Using dummy data.") # Keep console warnings for non-critical files
    if not os.path.exists(SHAP_VALUES_FILE):
         print(f"Warning: {SHAP_VALUES_FILE} not found. Using dummy data.") # Keep console warnings

    main()