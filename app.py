import streamlit as st
from models import App

# Streamlit UI setup
st.title("SkyNet - Sky is the limit")
st.header("How can I help you today?")

if _name_ == "_main_":
    app = App()
    app()