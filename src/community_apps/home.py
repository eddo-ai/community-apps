import streamlit as st

st.title("Welcome to the community apps showcase! 🚀")
st.caption(
    "Apps for teachers, coaches, and instructional leaders, co-created by educators and the Eddo Learning team."
)

# if not st.experimental_user.is_logged_in:
#     st.button("Log in", on_click=st.login)

with st.container(border=True):
    st.header("Student Work Analysis")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(
            "images/20250416_1300_Student Work Analysis Cover_remix_01js02x7zhftb8h81bt9f068nb.png"
        )
    with col2:
        st.write("Analyze student work with AI-powered insights and feedback.")
        st.write(
            "This app allows you to upload student responses and get instant summaries and insights across the dataset."
        )
        st.write("A collaboration with the Wauwatosa School District science team.")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.page_link(
                "apps/analyze_student_work.py",
                label="Try it out",
                icon=":material/open_in_new:",
            )
        with col2:
            st.page_link(
                "https://eddolearning.com/analyzing-student-work/",
                label="See the full recap",
                icon=":material/open_in_new:",
            )
