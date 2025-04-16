from langchain import hub
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import streamlit as st
import asyncio
import base64
import pandas as pd
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Enable detailed logging for httpx (which handles the network requests)
logging.getLogger("httpx").setLevel(logging.DEBUG)

st.title("Transcribe Images")

st.write(
    """
This tool transcribes images of hand-written notes into text.
"""
)

with st.expander("View prompt"):
    prompt = hub.pull("transcribe_student_work")
    st.write("This prompt uses the following templates:")

    # Get the system message template
    system_msg = prompt.messages[0]
    st.write("**System Message:**")
    st.write(system_msg.prompt.template)

    # Get the human message template
    st.write("\n**Human Message:**")
    human_msg = prompt.messages[1]
    for template in human_msg.prompt:
        if template.template:  # Only show non-empty templates
            st.write(template.template)


def convert_image_to_base64(uploaded_file):
    # Read the BytesIO buffer
    bytes_data = uploaded_file.getvalue()

    # Get the mime type from the uploaded file
    file_type = uploaded_file.type

    # Encode to base64
    base64_str = base64.b64encode(bytes_data).decode()

    # Create the base64 string with mime type
    return f"data:{file_type};base64,{base64_str}"


def clean_content(text):
    """Clean text content by removing newlines and extra whitespace"""
    if not text:
        return ""
    # Replace newlines with spaces and remove extra whitespace
    return " ".join(text.replace("\n", " ").split())


def save_and_display_results(results):
    """Save results to CSV and display them in a table"""
    if not results:
        return

    # Create data directory if it doesn't exist
    data_dir = "data/transcriptions"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = f"{data_dir}/transcriptions.csv"

    # Process all results
    rows = []
    for filename, result in results:
        if not result.get("is_orientation_upright"):
            st.error(
                f"Image {filename} is not upright. Please rotate it and try again."
            )
            continue

        for response in result.get("responses", []):
            row = {
                "timestamp": datetime.now(),
                "filename": filename,
                "prompt": clean_content(response.get("prompt")),
                "content": clean_content(response.get("content")),
            }
            rows.append(row)

    if rows:
        # Save to CSV
        df = pd.DataFrame(rows)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # Display results
        st.subheader("Transcription Results")
        st.dataframe(
            df.drop("timestamp", axis=1),  # Don't show timestamp in display
            column_config={
                "filename": st.column_config.TextColumn("Filename", width="medium"),
                "prompt": st.column_config.TextColumn("Prompt", width="medium"),
                "content": st.column_config.TextColumn("Content", width="large"),
            },
            hide_index=True,
        )
    return results


@st.cache_data(show_spinner=False)
async def transcribe_single_image(image_data: str, model) -> dict:
    """Transcribe a single image using the model. This function is cached."""
    chain = hub.pull("transcribe_student_work") | model
    return await chain.ainvoke(image_data)


async def transcribe_images(uploaded_files):
    if len(uploaded_files) == 0:
        st.error("No images uploaded. Please upload some images first.")
        return []

    MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4")

    # Try Azure OpenAI first, fall back to OpenAI
    azure_key = st.secrets.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        model = AzureChatOpenAI(
            api_key=azure_key,
            azure_endpoint=st.secrets.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=st.secrets.get("AZURE_OPENAI_DEPLOYMENT"),
            model_name=MODEL_NAME,
            temperature=0,  # Add temperature=0 for more consistent results
        )
    else:
        openai_key = st.secrets.get("OPENAI_API_KEY")
        if not openai_key:
            st.error(
                "No OpenAI API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY in secrets."
            )
            return []
        model = ChatOpenAI(
            api_key=openai_key,
            model_name=MODEL_NAME,
            temperature=0,  # Add temperature=0 for more consistent results
        )

    # Process images one by one using the cached function
    results = []
    for file in uploaded_files:
        base64_image = convert_image_to_base64(file)
        result = await transcribe_single_image(base64_image, model)
        results.append((file.name, result))

    return save_and_display_results(results)


# Transcribe the images
with st.form(key="upload_images", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if st.form_submit_button("Transcribe"):
        with st.status("Transcribing images...") as status:
            try:
                results = asyncio.run(transcribe_images(uploaded_files))
                if results:
                    status.update(label="Transcription complete!", state="complete")
                else:
                    status.update(label="No images were processed", state="error")
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")
                status.update(label="Transcription failed", state="error")
