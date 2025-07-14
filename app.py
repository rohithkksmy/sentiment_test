import streamlit as st
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# 🔐 Load API keys from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]

# ✅ Authenticate with OpenAI and Hugging Face
client = OpenAI(api_key=openai_api_key)
login(token=hf_token)

# ✅ Load and cache the Hugging Face sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_model = load_sentiment_model()

# ✅ Streamlit UI setup
st.set_page_config(page_title="GPT Chatbot + Sentiment", page_icon="🤖")
st.title("💬 GPT Chatbot with Real-Time Sentiment Analysis")

# ✅ Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

# ✅ Display full conversation history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Chat input field
user_input = st.chat_input("Say something...")

if user_input:
    # Add user message
    st.session_state.user_inputs.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # GPT-4 response
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

# ✅ Sentiment analysis button
if st.button("🔍 Analyze Overall Sentiment"):
    if st.session_state.user_inputs:
        with st.spinner("Analyzing sentiment..."):
            try:
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
                st.caption(f"Out of {total} messages — ✅ {positive} positive, ❌ {negative} negative")
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
    else:
        st.warning("No user messages to analyze.")
