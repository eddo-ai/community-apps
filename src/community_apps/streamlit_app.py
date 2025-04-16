import streamlit as st

# Set consistent page config
st.set_page_config(
    page_title="Community Apps by Eddo Learning", page_icon="🧑‍💻", layout="wide"
)


# Import pages
home_page = st.Page("home.py", title="Home", icon="🏠", default=True)
apps = [
    st.Page("apps/analyze_student_work.py", title="Analyze Student Work", icon="📝"),
]


pg = st.navigation(
        {
            "Main": [home_page],
            "Apps": apps,
        }
    )

# Run the selected page
pg.run()
