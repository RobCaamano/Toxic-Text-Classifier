import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)
from transformers import pipeline

st.title("Toxic Tweet Classifier")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

text = ""
submit = False
model_name = ""
col1, col2, col3 = st.columns([2,1,1])

with st.container():
    model_name = st.selectbox(
        "Select the model you want to use below.",
        ("RobCaamano/toxicity",),
    )
    submit = st.button("Submit", type="primary")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

with col1:
    st.subheader("Tweet")
    text = st.text_area("Input text", demo, height=275)

input = tokenizer(text, return_tensors="tf")

if submit:
    results = dict(d.values() for d in clf(text)[0])
    classes = {k: results[k] for k in results.keys() if not k == "toxic"}

    max_class = max(classes, key=classes.get)
    probability = classes[max_class]

    result_df = pd.DataFrame({
        'Classification': [max_class],
        'Probability': [probability],
        'Toxic': ['Yes' if results['toxic'] >= 0.5 else 'No']
    })

    st.table(result_df)

    expander = st.expander("Raw output")
    expander.write(results)
