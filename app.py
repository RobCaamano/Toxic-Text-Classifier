import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, pipeline
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)

st.title("Classifier")

demo_options = {
    "Non-toxic": "Had a wonderful weekend at the park. Enjoyed the beautiful weather!",
    "Severe-toxic": "WIP",
    "Obscene": "I don't give a fuck about your opinion",
    "Threat": "WIP",
    "Insult": "Are you always this incompetent?",
    "Identity Hate": "WIP",
}

selected_demo = st.selectbox("Demos", options=list(demo_options.keys()))
text = st.text_area("Input text", demo_options[selected_demo], height=250)

submit = False
model_name = ""

model_mapping = {
    "Toxicity": "RobCaamano/toxicity",
    "Toxicity 2": "RobCaamano/toxicity_distilbert",
    "DistilBERT Base Uncased (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
}

with st.container():
    selected_model_display = st.selectbox(
        "Select Model",
        options=list(model_mapping.keys())
    )
    model_name = model_mapping[selected_model_display]
    submit = st.button("Submit", type="primary")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if submit:
    results = dict(d.values() for d in clf(text)[0])

    if model_name in ["RobCaamano/toxicity", "RobCaamano/toxicity_distilbert"]:
        classes = {k: results[k] for k in results.keys() if not k == "toxic"}

        max_class = max(classes, key=classes.get)
        probability = classes[max_class]

        if results['toxic'] >= 0.5:
            result_df = pd.DataFrame({
                'Toxic': 'Yes',
                'Toxicity Class': [max_class],
                'Probability': [probability]
            }, index=[0])
        else:
            result_df = pd.DataFrame({
                'Toxic': 'No',
                'Toxicity Class': 'This text is not toxic',
            }, index=[0])

    elif model_name == "distilbert-base-uncased-finetuned-sst-2-english":
        result = max(results, key=results.get)
        probability = results[result]

        result_df = pd.DataFrame({
            'Result': [result],
            'Probability': [probability],
        }, index=[0])

    st.table(result_df)

    expander = st.expander("View Raw output")
    expander.write(results)
