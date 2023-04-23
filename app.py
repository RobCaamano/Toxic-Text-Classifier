import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)
from transformers import pipeline

st.title("Detecting Toxic Tweets")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

text = st.text_area("Input text", demo, height=250)

# Add a drop-down menu for model selection
model_options = {
    "DistilBERT Base Uncased (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "Fine-tuned Toxicity Model": "https://huggingface.co/RobCaamano/toxicity_distilbert",
}
selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

mod_name = model_options[selected_model]

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)
clf = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if st.button("Submit", type="primary"):
    results = clf(text)[0]
    max_class = max(results, key=lambda x: x["score"])
    tweet_portion = text[:50] + "..." if len(text) > 50 else text
    
    # Create and display the table
    df = pd.DataFrame(
        {
            "Tweet (portion)": [tweet_portion],
            "Highest Toxicity Class": [max_class["label"]],
            "Probability": [max_class["score"]],
        }
    )
    st.table(df)
