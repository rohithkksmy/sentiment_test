import streamlit as st
import openai
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Hugging Face auth
login(token=st.secrets["HF_TOKEN"])

@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_model = load_sentiment_model()

# UI setup
st.set_page_config(page_title="GPT Chatbot + Sentiment", page_icon="🤖")
st.title("💬 GPT Chatbot with Sentiment Analysis")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.user_inputs.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# Sentiment analysis
if st.button("🔍 Analyze Overall Sentiment"):
    if st.session_state.user_inputs:
        with st.spinner("Analyzing..."):
            results = sentiment_model(st.session_state.user_inputs)
            positive = sum(1 for r in results if r["label"] == "POSITIVE")
            negative = sum(1 for r in results if r["label"] == "NEGATIVE")
            total = len(results)

            if positive > negative:
                sentiment = "🙂 Positive"
            elif negative > positive:
                sentiment = "🙁 Negative"
            else:
                sentiment = "😐 Neutral"

            st.success(f"**Overall Sentiment:** {sentiment}")
            st.caption(f"(Out of {total} messages — ✅ {positive} positive, ❌ {negative} negative)")
    else:
        st.warning("No user messages to analyze.")
