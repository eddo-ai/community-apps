import streamlit as st

# Set consistent page config
st.set_page_config(
    page_title="Community Apps",
    page_icon="🧑‍💻",
    layout="wide"
)


# Import pages
home_page = st.Page("home.py", title="Home", icon="🏠", default=True)
apps = [
    st.Page("apps/transcribe_images.py", title="Transcribe Images", icon="📸"),
    st.Page("apps/assessment_feedback.py", title="Assessment Feedback", icon="📝"),
]


if st.experimental_user.is_logged_in is True:
    # Show navigation and pages for logged in users
    pg = st.navigation({
        "Main": [home_page],
        "Apps": apps,
    })
    with st.sidebar:
        st.text(st.experimental_user.get("email", ""))
        col1, col2 = st.columns([1, 3])
        with col1:
            if picture := st.experimental_user.get("picture"):
                st.image(picture)
        with col2:
            if name := st.experimental_user.get("name"):
                st.write(f"Welcome, {name}!")
            else:
                st.write("Welcome!")
            st.button("Log out", on_click=st.logout)
        st.divider()
else:
    pg = st.navigation({
        "Main": [home_page],
    })

# Run the selected page
pg.run()

    