import streamlit as st
import pandas as pd
import numpy as np
from ml import pred


st.title('Earthquake Prediction')
lat = st.number_input("Enter Lattitude ", 0.0000, 90.0000, 28.7041, 0.0001)
lon = st.number_input("Enter Longitude ", 0.0000, 90.0000, 77.1025, 0.0001)

st.text(pred(lat, lon)[0])