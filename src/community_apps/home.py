import streamlit as st

st.title("Welcome to Community Apps! 🚀")

st.write("This is the home page of the Community Apps. Here you can find all the apps that are available to you.")

if not st.experimental_user.is_logged_in:
    st.button("Log in", on_click=st.login)

