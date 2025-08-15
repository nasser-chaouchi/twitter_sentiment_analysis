import streamlit as st
import pickle
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download
import pandas as pd


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
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(
        "nasserCha/twitter_sentiment_analysis",
        subfolder="models/bert_sentiment_model"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "nasserCha/twitter_sentiment_analysis",
        subfolder="models/bert_sentiment_model"
    )
    return model, tokenizer


nb_model = load_nb_model()
bert_model, bert_tokenizer = load_bert_model()

label_map = {0: "Negative", 1: "Positive"}


st.title("Twitter Sentiment Analysis")
st.markdown("""
This application compares two sentiment analysis models trained on the Sentiment140 dataset:
- **TF-IDF + Multinomial Naive Bayes**: Fast and interpretable, but less accurate.
- **BERT Fine-Tuned**: More accurate, but slower and more resource-intensive.
""")


st.subheader("Model Performance Comparison")

results = {
    "Model": ["TF-IDF + Naive Bayes", "BERT Fine-Tuned"],
    "Accuracy": [0.77, 0.83],
    "F1-score": [0.77, 0.83],
    "Notes": [
        "Fast, interpretable, but limited in context understanding",
        "Slower, resource-intensive, but better at capturing nuanced sentiment"
    ]
}

df_results = pd.DataFrame(results)
st.table(df_results)

st.markdown("""
**Conclusion:**  
While Naive Bayes is faster and easier to interpret, BERT offers significantly higher accuracy and better handles contextual and nuanced sentiment, making it more suitable for production use.
""")


st.subheader("Test a Tweet")
user_input = st.text_area("Tweet text", "")

model_choice = st.selectbox("Choose a model", ["Naive Bayes", "BERT"])

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet text.")
    else:
        if model_choice == "Naive Bayes":
            pred_class = nb_model.predict([user_input])[0]
            st.write(f"**Predicted Sentiment:** {label_map[pred_class]}")
        
        elif model_choice == "BERT":
            inputs = bert_tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            st.write(f"**Predicted Sentiment:** {label_map[pred_class]} ({confidence:.2%} confidence)")

st.markdown("---")
st.markdown("**Author:** Nasser Chaouchi - Project for portfolio")
