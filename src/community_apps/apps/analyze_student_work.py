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


st.title("Analyze Student Work")

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
        MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")

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
    default_data_path = os.path.join("data", "feedback-sample.csv")
    try:
        df = pd.read_csv(default_data_path)

        # Add data preview
        with st.expander("Sample Student Responses", expanded=True):
            st.dataframe(df, hide_index=True)

    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        st.stop()

    if df is not None:
        st.header("3. Analyze Student Work Samples")
DISCUSSION_GUIDE_PROMPT = """
The provided dataset includes responses to two prompts about "zombie fires" and their environmental implications. Each entry contains student responses, assigned scores, feedback for improvement, and noted strengths. Here's a summary of the analysis:

### Observations and Patterns:
1. **Prompt A: How Zombie Fires Burn Under Ice**
   - Common strengths:
     - Clear explanations of the role of peat formation and arctic conditions in sustaining fires.
     - Some responses effectively link past arctic conditions with current phenomena.
   - Common areas for improvement:
     - A need for better organization and clarity in responses.
     - Inclusion of more specific details or scientific concepts.

2. **Prompt B: Impact of Rising Temperatures**
   - Common strengths:
     - Use of predictive reasoning about environmental changes.
     - Some students integrated zombie fires into their explanations effectively.
   - Common areas for improvement:
     - Responses often lack specific details or strong connections to the provided data.
     - Some responses need better logical flow or deeper analysis.

3. **Scoring Trends:**
   - Scores range from 0 to 4, with higher-scoring responses being more detailed and well-structured.
   - Lower scores frequently correlate with disorganized responses or incomplete reasoning.

### Recommendations for Improvement:
- Encourage students to:
  - Use the provided sentence starters to structure their responses.
  - Integrate specific evidence or examples to support claims.
  - Clearly articulate the connection between cause and effect in their explanations.

- Provide examples of high-scoring responses to illustrate expectations.
- Focus on scaffolding support for lower-performing students, particularly on organizing ideas and incorporating details.

This analysis highlights areas where students excel and where additional guidance can elevate their performance. Let me know if you'd like further breakdowns or visualizations of the data!
"""
DISCUSSION_GUIDE_URL = "https://chatgpt.com/share/68000e9d-12e0-8012-980f-97696ecded35"
with st.expander("Discussion Guide Prompt", expanded=False):
    st.markdown(DISCUSSION_GUIDE_PROMPT)
    st.link_button(
        "View in ChatGPT", DISCUSSION_GUIDE_URL, icon=":material/open_in_new:"
    )


def get_analysis_chain():
    """Initialize and return the LangChain chain for analyzing student responses."""
    if "analysis_chain" not in st.session_state:
        # Get model name from secrets
        MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")

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

        prompt = hub.pull("hey-aw/analyze_student_work")

        st.session_state.analysis_chain = prompt | model

    return st.session_state.analysis_chain


st.header("4. Summary and Analysis")
st.subheader("Example")
with st.chat_message("teacher", avatar="user"):
    st.markdown(
        """
    analyze these samples and summarize
    """
    )
with st.chat_message("assistant", avatar="assistant"):
    st.markdown(DISCUSSION_GUIDE_PROMPT)

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
                                "dataset": df,
                                "question": prompt,
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
