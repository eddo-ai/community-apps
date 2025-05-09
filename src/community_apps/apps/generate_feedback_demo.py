# assessment_feedback.py

import streamlit as st
import pandas as pd
from langchain import hub
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import asyncio
import os
import numpy as np  # Add numpy import for random selection
from textwrap import fill


def display_prompt(prompt):
    """Display a LangChain prompt in a readable format using Streamlit chat messages.

    Args:
        prompt: A LangChain prompt object
    """

    # Display messages in a chat-like format
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Feedback Prompt")
    for msg in prompt.messages:
        with st.container(border=True):
            template = msg.prompt.template
            st.markdown(template)

    with col2:
        # Display metadata and hub link if present
        if hasattr(prompt, "metadata") and prompt.metadata:
            owner = prompt.metadata.get("lc_hub_owner")
            repo = prompt.metadata.get("lc_hub_repo")
            if owner and repo:
                hub_url = f"https://smith.langchain.com/hub/{owner}/{repo}"
                st.link_button(
                    "View full template", hub_url, icon=":material/open_in_new:"
                )


st.title("Demo: Generate Feedback on Student Work")

# Initialize session state variables
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "instructions" not in st.session_state:
    st.session_state.instructions = ""
if "previous_selection" not in st.session_state:
    st.session_state.previous_selection = []
if "random_seed" not in st.session_state:
    st.session_state.random_seed = np.random.randint(0, 10000)

# Default instructions
DEFAULT_INSTRUCTIONS = """
# L6 Zombie Fire Explanation

Summative To study changing fires in the Arctic, scientists took pictures of fires from space, using special cameras. The first image (top-left), acquired in September 2015, shows the burn scar from the Soda Creek Fire, which scorched nearly 17,000 acres in southwest Alaska near the Kuskokwim River. The fire was never completely extinguished before winter set in. In April 2016 (top-right), the fire continued to smolder in the peat under a layer of snow. When the snow finally melted in late May (bottom-left), the additional heat and oxygen caused flames to re-emerge and quickly spread.

| Instructions: Use evidence from your investigations in class to explain both questions below:   (A) How was there enough matter and energy in the system for a zombie fire to burn under ice?    (B) What will happen in the future if temperatures continue to rise? |  |
| :---- | :---- |
| **Sentence starters:**    The components of the zombie fire system are…    Zombie fires can burn under ice because…     The peat formed by the process of….    If the temperatures of the Earth continue to rise…  | Energy flows through the system from… to… through the process of…   The matter in the system transforms from… to … through the process of…  In the past, the arctic was different than today because….  |
| **Key vocabulary to include:**    Photosynthesis    Cellular respiration / Decomposition | Oxygen Temperature Energy |

| 4 Extending | 3 Proficient | 2 Approaching | 1 Beginning |
| :---- | :---- | :---- | :---- |
| Constructs an explanation about zombie fires based on empirical evidence and makes specific connections to multiple disciplinary ideas. Cites specific evidence from multiple class activities.   | Constructs an explanation about zombie fires based on empirical evidence and makes a connection to multiple disciplinary ideas. Cites specific evidence from class.  (Don't use words like more or a lot.) | Constructs an explanation about zombie fires based on empirical evidence and begins to make a connection to multiple disciplinary ideas. References evidence from class (without citing specific data).  | Constructs an explanation (**including interactions**) to describe zombie fires with **minimal** reference to empirical evidence.  |

| Check | Attribute |
| ----- | ----- |
|  | Did I make a claim about matter and energy in the zombie fire system?  Does my claim discuss energy both flowing into and out of the zombie fire system?  |
|  | Did I include specific evidence from activities we did in class? Burning fuel samples demonstration Yeast/Decomposition Lab Elodea/Earth's Tilt lab Progress Tracker |
|  | Did I explain how my evidence supports my claim?  Did I apply multiple scientific ideas to my explanation? Carbon cycle Cellular respiration Photosynthesis Decomposition Solar radiation |
|  | Did I make a prediction about the future?  |
|  | Did I reference evidence from this unit to support my prediction? |
|  | Is my writing legible?  Is my writing concise? |

"""

prompt = st.session_state.get("prompt", hub.pull("hey-aw/generate_student_feedback"))
st.session_state.prompt = prompt


def get_chain():
    """Initialize and return the LangChain chain."""
    if "chain" not in st.session_state:
        prompt = st.session_state.get(
            "prompt", hub.pull("hey-aw/generate_student_feedback")
        )
        st.session_state.prompt = prompt

        # Get model name from secrets
        MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4")

        # Try Azure OpenAI first, fall back to OpenAI
        azure_key = st.secrets.get("AZURE_OPENAI_API_KEY")
        if azure_key:
            model = AzureChatOpenAI(
                api_key=azure_key,
                azure_endpoint=st.secrets.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=st.secrets.get("AZURE_OPENAI_DEPLOYMENT"),
                model_name=MODEL_NAME,
                temperature=0,
            )
        else:
            openai_key = st.secrets.get("OPENAI_API_KEY")
            if not openai_key:
                st.error(
                    "No OpenAI API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY in secrets."
                )
                return None
            model = ChatOpenAI(
                api_key=openai_key,
                model_name=MODEL_NAME,
                temperature=0,
            )

        st.session_state.chain = prompt | model
    return st.session_state.chain


async def generate_feedback(student_response, instructions):
    """Generate feedback for a single student response."""
    chain = get_chain()
    if chain is None:
        return None

    try:
        result = await chain.ainvoke(
            {
                "student_response": student_response,
                "instructions": instructions,
            }
        )
        return result
    except Exception as e:
        st.error(f"Error generating feedback: {str(e)}")
        return None


# Instructions setup section
st.header("1. Instructions and Assessment Criteria")

# Display the instructions in a container
with st.expander("Instructions", expanded=True):
    st.markdown(DEFAULT_INSTRUCTIONS)

# Store instructions in session state
st.session_state.instructions = DEFAULT_INSTRUCTIONS

# Process responses section
if not st.session_state.get("instructions"):
    st.stop()
else:
    st.header("2. Student Work Samples")

    # Load default data
    default_data_path = os.path.join("data", "transcriptions.csv")

    try:
        df = pd.read_csv(default_data_path)

        # Add data preview
        with st.expander("Sample Student Responses", expanded=True):
            st.dataframe(df, hide_index=True)

    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        st.stop()

    if df is not None:
        st.header("3. Generate Feedback")

        # Generate random student response
        np.random.seed(st.session_state.random_seed)
        random_idx = np.random.choice(len(df))
        student_response = df.iloc[random_idx]["content"]

        # Create two columns for side-by-side view
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Random Student Response")
            # Format the response with proper paragraphs
            paragraphs = str(student_response).split("\n")
            formatted_paragraphs = []
            for p in paragraphs:
                if p.strip():  # Skip empty paragraphs
                    wrapped = fill(
                        p.strip(),
                        width=80,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    formatted_paragraphs.append(wrapped)
            st.markdown("\n\n".join(formatted_paragraphs))

        with col2:
            st.markdown("#### Generate Feedback")
            # Preview the prompt
            with st.expander("View chat prompt", expanded=False):
                display_prompt(st.session_state.prompt)

            if st.button("Generate", type="primary"):
                with st.spinner("Generating feedback..."):
                    try:
                        feedback = asyncio.run(
                            generate_feedback(
                                student_response, st.session_state.instructions
                            )
                        )

                        if feedback:
                            # Display strengths
                            if "strengths" in feedback:
                                st.markdown("##### 💪 Strengths")
                                st.markdown(feedback["strengths"])
                                st.markdown("")  # Add spacing

                            # Display areas for improvement
                            if "improvement" in feedback:
                                st.markdown("##### 🎯 Areas for Improvement")
                                st.markdown(feedback["improvement"])
                                st.markdown("")  # Add spacing

                            # Display any additional feedback fields
                            for key, value in feedback.items():
                                if key not in [
                                    "strengths",
                                    "improvement",
                                    "student_response",
                                    "instructions",
                                ]:
                                    st.markdown(
                                        f"##### {key.replace('_', ' ').title()}"
                                    )
                                    st.markdown(value)
                                    st.markdown("")  # Add spacing

                        else:
                            st.error("Failed to generate feedback")
                    except Exception as e:
                        st.error(f"Error generating feedback: {str(e)}")

        # Add a "Try Another" button to get a new random response
        if st.button("Try Another Response"):
            st.session_state.random_seed = np.random.randint(0, 10000)
            st.rerun()


def get_analysis_chain():
    """Initialize and return the LangChain chain for analyzing student responses."""
    if "analysis_chain" not in st.session_state:
        # Get model name from secrets
        MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4")

        # Try Azure OpenAI first, fall back to OpenAI
        azure_key = st.secrets.get("AZURE_OPENAI_API_KEY")
        if azure_key:
            model = AzureChatOpenAI(
                api_key=azure_key,
                azure_endpoint=st.secrets.get("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=st.secrets.get("AZURE_OPENAI_DEPLOYMENT"),
                model_name=MODEL_NAME,
                temperature=0,
            )
        else:
            openai_key = st.secrets.get("OPENAI_API_KEY")
            if not openai_key:
                st.error(
                    "No OpenAI API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY in secrets."
                )
                return None
            model = ChatOpenAI(
                api_key=openai_key,
                model_name=MODEL_NAME,
                temperature=0,
            )

        from langchain.prompts import ChatPromptTemplate

        template = """You are an expert in analyzing student responses and identifying patterns.
        
        Analyze the following student responses to answer this question: {question}

        Context: These are responses to a question about zombie fires in the Arctic, where students were asked to explain:
        A) How there was enough matter and energy in the system for a zombie fire to burn under ice
        B) What will happen in the future if temperatures continue to rise

        Student Responses:
        {responses}

        Provide a clear, concise analysis that directly answers the question. Base your analysis only on the actual student responses provided.
        Focus on patterns, trends, and specific examples from the responses to support your analysis.
        """

        prompt = ChatPromptTemplate.from_template(template)
        st.session_state.analysis_chain = prompt | model

    return st.session_state.analysis_chain


st.header("4. Analysis")

# Initialize messages in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me questions about the student responses! For example:\n- What common misconceptions do students have?\n- How many students mentioned photosynthesis?\n- What's the average response length?",
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the student responses"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing responses..."):
            try:
                chain = get_analysis_chain()
                if chain:
                    result = asyncio.run(
                        chain.ainvoke(
                            {
                                "question": prompt,
                                "responses": "\n\n".join(df["content"].tolist()),
                            }
                        )
                    )

                    response = result.content
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                else:
                    st.error(
                        "Chat model not available. Please check your API configuration."
                    )

            except Exception as e:
                st.error(f"Error analyzing responses: {str(e)}")
