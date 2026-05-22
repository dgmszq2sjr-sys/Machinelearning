#importing necessary libraries for the application
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import io
# Set up a clean web page structure
st.set_page_config(page_title="Iris Species Predictor", layout="centered")
st.title("Iris Species Predictor")
# STEP 1: LOADing dataset from iris.csv file
print("Loading dataset from the csv file(comma separated values)...")
df = pd.read_csv('iris.csv')
# Cleaning the column headers automatically to avoid fixes typos or hidden spaces
df.columns = df.columns.str.strip().str.lower()
# Clean species names for simpler legend display
df['species'] = df['species'].str.replace('Iris-', '').str.capitalize()
# Clean species names for simpler legend display
df['species'] = df['species'].str.replace('Iris-', '').str.capitalize()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# STEP 2: BUILD SLIDER INTERFACE (GRID)
# Sliders are placed at the bottom, so we capture their variables first
st.write("---")
slider_col1, slider_col2 = st.columns(2)
with slider_col1:
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.8, step=0.1)
    pl = st.slider("Petal Length", 1.0, 7.0, 3.8, step=0.1)
with slider_col2:
    sw = st.slider("Sepal Width", 2.0, 4.5, 3.0, step=0.1)
    pw = st.slider("Petal Width", 0.1, 2.5, 1.2, step=0.1)
# RUN LIVE ML PREDICTION
input_features = np.array([[sl, sw, pl, pw]])
prediction = model.predict(input_features)[0]
probabilities = model.predict_proba(input_features)[0]
confidence = max(probabilities) * 100
# GENERATE THE GRAPH WITH LIVE DOT
# Color map matching the dashboard styles
color_map = {'Setosa': 'blue', 'Versicolor': 'green', 'Virginica': 'orange'}
# Base scatter plot of historical dataset points
fig = px.scatter(
    df, 
    x='petal_length', 
    y='petal_width', 
    color='species',
    color_discrete_map=color_map,
    labels={'petal_length': 'Petal Length (cm)', 'petal_width': 'Petal Width (cm)'},
    range_x=[0.5, 7.5],
    range_y=[-0.2, 3.2]
)
# Overlay the "Current Selection" dot trace dynamically
fig.add_trace(
    go.Scatter(
        x=[pl], 
        y=[pw],
        mode='markers+text',
        name='Current',
        text=['Current Selection'],
        textposition='top center',
        marker=dict(color='white', size=15, line=dict(color='white', width=2)),
        showlegend=True
    )
)
# Apply dark theme styling to match the image environment
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=20, r=20, t=60, b=20)
)
# Render plot to screen above the sliders
st.plotly_chart(fig, use_container_width=True)
# STEP 5: METRIC DISPLAY BLOCK
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.markdown("<center style='color:#aaa; font-size:14px;'>Predicted Species</center>", unsafe_allow_html=True)
    st.markdown(f"<center style='font-size:22px; font-weight:bold; font-family:monospace;'>{prediction}</center>", unsafe_allow_html=True)
with metric_col2:
    st.markdown("<center style='color:#aaa; font-size:14px;'>Confidence</center>", unsafe_allow_html=True)
    st.markdown(f"<center style='font-size:22px; font-weight:bold; font-family:monospace;'>{confidence:.0f}%</center>", unsafe_allow_html=True)