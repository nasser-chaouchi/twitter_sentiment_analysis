import streamlit as st
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pandas as pd
from typing import Tuple


st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon=None,
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* Typography */
html, body, [class*="css"]  {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
}
/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #EAECF0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    margin-bottom: 16px;
}
.card-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 8px;
    color: #111827;
}
.subtle {
    color: #6B7280;
}
/* Prediction badge */
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
}
.badge-positive { background: #E8FAF1; color: #067647; border: 1px solid #C6F6D9; }
.badge-negative { background: #FEF3F2; color: #B42318; border: 1px solid #FEE4E2; }
/* Confidence bar container */
.confbar {
    height: 10px;
    background: #F3F4F6;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}
.confbar-fill {
    height: 100%;
    background: #2563EB;
}
.small {
    font-size: 0.9rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_nb_model():
    model_path = hf_hub_download(
        repo_id="nasserCha/twitter_sentiment_analysis",
        filename="models/nb_tfidf_pipeline.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_bert_model() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(
        "nasserCha/twitter_sentiment_analysis",
        subfolder="models/bert_sentiment_model"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nasserCha/twitter_sentiment_analysis",
        subfolder="models/bert_sentiment_model"
    )
    return model, tokenizer

@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


nb_model = load_nb_model()
bert_model, bert_tokenizer = load_bert_model()
device = get_device()
bert_model.to(device)
bert_model.eval()

LABEL_MAP = {0: "Negative", 1: "Positive"}


with st.sidebar:
    st.markdown("### Settings")
    model_choice = st.selectbox("Model", ["BERT", "Naive Bayes"], index=0)
    max_length = st.slider("Max token length (BERT)", 32, 256, 96, step=16)
    show_prob = st.checkbox("Show class probabilities", value=True)
    st.markdown("---")
    st.markdown(
        "Models and dataset are hosted on Hugging Face Hub.\n"
        "[Repository](https://huggingface.co/nasserCha/twitter_sentiment_analysis)",
        help="The app downloads the artifacts from the Hub at runtime."
    )


st.title("Twitter Sentiment Analysis")
st.markdown(
    "Binary sentiment classification trained on the Sentiment140 dataset. "
    "This application provides a clean comparison between a TF-IDF + Naive Bayes baseline and a fine-tuned BERT model."
)


reported = pd.DataFrame({
    "Model": ["TF-IDF + Naive Bayes", "BERT (fine-tuned)"],
    "Accuracy": [0.78, 0.91],
    "F1 (macro)": [0.77, 0.91]
})

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown('<div class="card"><div class="card-title">Naive Bayes (reported)</div>', unsafe_allow_html=True)
    st.metric(label="Accuracy", value=f"{reported.loc[0, 'Accuracy']:.2f}")
    st.metric(label="F1 (macro)", value=f"{reported.loc[0, 'F1 (macro)']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><div class="card-title">BERT (reported)</div>', unsafe_allow_html=True)
    st.metric(label="Accuracy", value=f"{reported.loc[1, 'Accuracy']:.2f}")
    st.metric(label="F1 (macro)", value=f"{reported.loc[1, 'F1 (macro)']:.2f}")
    st.markdown(f'<span class="subtle small">Device: {device.type.upper()}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><div class="card-title">Summary</div>', unsafe_allow_html=True)
    st.write(
        "Naive Bayes provides a fast, interpretable baseline. "
        "BERT offers higher accuracy and better handling of contextual language."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Inference</div>', unsafe_allow_html=True)

examples = {
    "Positive example": "I really like this new update, it works perfectly.",
    "Negative example": "This is the worst service I have ever used.",
    "Neutral wording with positive intent": "Not bad at all, actually quite impressed."
}


if "tweet_text" not in st.session_state:
    st.session_state["tweet_text"] = ""
if "example_key" not in st.session_state:
    st.session_state["example_key"] = list(examples.keys())[0]


def use_example_callback():
    st.session_state["tweet_text"] = examples[st.session_state["example_key"]]


col_a, col_b = st.columns([3, 1])

with col_b:
    st.selectbox("Examples", list(examples.keys()), key="example_key")
    st.button("Use example", on_click=use_example_callback)

with col_a:
    text = st.text_area(
        "Tweet text",
        height=120,
        placeholder="Type or paste a tweet...",
        value=st.session_state["tweet_text"],
        key="tweet_text"
    )

def predict_nb(s: str) -> Tuple[int, float]:
    pred = nb_model.predict([s])[0]
    try:
        proba = nb_model.predict_proba([s])[0][pred]
    except Exception:
        proba = 1.0
    return int(pred), float(proba)

def predict_bert(s: str, max_len: int) -> Tuple[int, float]:
    inputs = bert_tokenizer(
        s,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    pred = int(torch.argmax(probs, dim=1).item())
    conf = float(probs[0][pred].item())
    return pred, conf

submit = st.button("Predict")

if submit:
    if not text or not text.strip():
        st.warning("Please provide a non-empty input.")
    else:
        if model_choice == "Naive Bayes":
            y, conf = predict_nb(text)
        else:
            y, conf = predict_bert(text, max_length)

        label = LABEL_MAP[y]
        badge_class = "badge-positive" if y == 1 else "badge-negative"

        st.markdown(f'<span class="badge {badge_class}">Prediction: {label}</span>', unsafe_allow_html=True)
        if show_prob:
            st.markdown(f'<div class="subtle small">Confidence: {conf:.2%}</div>', unsafe_allow_html=True)
            # Confidence bar
            pct = max(0.0, min(conf, 1.0))
            st.markdown(
                f'<div class="confbar"><div class="confbar-fill" style="width:{pct*100:.1f}%"></div></div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "Author: **Nasser Chaouchi** &nbsp;&nbsp;|&nbsp;&nbsp; "
    "[LinkedIn](https://www.linkedin.com/in/nasser-chaouchi/) &nbsp;&nbsp;|&nbsp;&nbsp; "
    "[Hugging Face](https://huggingface.co/nasserCha/twitter_sentiment_analysis)"
)