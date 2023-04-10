import streamlit as st
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)
from transformers import pipeline

st.title("Detecting Toxic Tweets")

demo = """I'm so proud of myself for accomplishing my goals today. #motivation #success"""

text = st.text_area("Input text", demo, height=250)

mod_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if st.button("Submit", type="primary"):
    results = clf(text)[0]
    st.write(f"The sentiment is {results}.")
