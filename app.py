import streamlit as st
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# Load API keys from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]

# Initialize OpenAI client and login to Hugging Face
client = OpenAI(api_key=openai_api_key)
login(token=hf_token)

# Cache Hugging Face sentiment model loading
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_model = load_sentiment_model()

# Streamlit page config
st.set_page_config(page_title="GPT Chatbot + Sentiment", page_icon="🤖")
st.title("💬 GPT Chatbot with Real-Time Sentiment Analysis")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

# Display the full chat history (skip system message)
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Say something...")

if user_input:
    # Append user message to session state
    st.session_state.user_inputs.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get GPT-4 response
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

# Sentiment analysis button
if st.button("🔍 Analyze Overall Sentiment"):
    if st.session_state.user_inputs:
        with st.spinner("Analyzing sentiment..."):
            try:
                results = sentiment_model(st.session_state.user_inputs)

                # Calculate net sentiment score with confidence weighting
                net_score = 0
                for r in results:
                    if r["label"] == "POSITIVE":
                        net_score += r["score"]
                    else:  # NEGATIVE
                        net_score -= r["score"]

                total = len(results)

                # Define thresholds to avoid noise
                if net_score > 0.1:
                    sentiment = "🙂 Positive"
                elif net_score < -0.1:
                    sentiment = "🙁 Negative"
                else:
                    sentiment = "😐 Neutral"

                st.success(f"**Overall Sentiment:** {sentiment}")
                st.caption(f"Analyzed {total} messages — Net sentiment score: {net_score:.2f}")

            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
    else:
        st.warning("No user messages to analyze.")
