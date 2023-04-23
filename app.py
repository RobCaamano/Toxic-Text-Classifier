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
    "Fine-tuned Toxicity Model": "RobCaamano/toxicity_distilbert",
}
selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

mod_name = model_options[selected_model]

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)

# Update the id2label mapping for the fine-tuned model
if selected_model == "Fine-tuned Toxicity Model":
    toxicity_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    model.config.id2label = {i: toxicity_classes[i] for i in range(model.config.num_labels)}

clf = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if st.button("Submit", type="primary"):
    results = clf(text)[0]
    max_class = max(results, key=lambda x: x["score"])
    
    tweet_portion = text[:50] + "..." if len(text) > 50 else text

    # Create and display the table
    if selected_model == "Fine-tuned Toxicity Model":
        column_name = "Highest Toxicity Class"
    else:
        column_name = "Prediction"
    
    df = pd.DataFrame(
        {
            "Tweet (portion)": [tweet_portion],
            column_name: [max_class["label"]],
            "Probability": [max_class["score"]],
        }
    )
    st.table(df)
