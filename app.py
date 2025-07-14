import streamlit as st
import openai
from transformers import pipeline

# Load OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

# UI Setup
st.set_page_config(page_title="GPT Chatbot with Sentiment", page_icon="🤖")
st.title("💬 GPT Chatbot + Sentiment Analysis")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    # Store and display user message
    st.session_state.user_inputs.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display GPT response
    with st.chat_message("assistant"):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or "gpt-3.5-turbo"
                messages=st.session_state.messages
            )
            reply = response.choices[0].message["content"]
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error: {e}")

# Analyze sentiment
if st.button("🔍 Analyze Overall Sentiment"):
    if st.session_state.user_inputs:
        with st.spinner("Analyzing..."):
            results = sentiment_model(st.session_state.user_inputs)
            positive = sum(1 for r in results if r["label"] == "POSITIVE")
            negative = sum(1 for r in results if r["label"] == "NEGATIVE")
            total = len(results)

            # Final sentiment decision
            if positive > negative:
                overall = "🙂 Positive"
            elif negative > positive:
                overall = "🙁 Negative"
            else:
                overall = "😐 Neutral"

            st.success(f"**Overall Sentiment:** {overall}")
            st.caption(f"(Based on {total} messages: {positive} positive, {negative} negative)")
    else:
        st.warning("No user messages to analyze.")
