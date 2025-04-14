import streamlit as st

if not st.experimental_user.is_logged_in:
    st.button("Log in with Google", on_click=st.login)
    st.stop()

st.button("Log out", on_click=st.logout)
if st.experimental_user.get("name"):
    st.markdown(f"Welcome! {st.experimental_user.get('name')}")
else:
    st.markdown("Welcome!")
    
with st.sidebar:
    st.write(st.experimental_user.get("email"))
