import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, pipeline
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)

st.title("Classifier")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

demo_options = {
    "non-toxic": "Had a wonderful weekend at the park. Enjoyed the beautiful weather!",
    "toxic": "WIP",
    "severe_toxic": "WIP",
    "obscene": "I don't give a fuck about your opinion",
    "threat": "WIP",
    "insult": "Are you always this incompetent?",
    "identity_hate": "WIP",
}

selected_demo = st.selectbox("Demos", options=list(demo_options.keys()))
text = st.text_area("Input text", demo_options[selected_demo] if selected_demo else demo, height=250)

submit = False
model_name = ""

with st.container():
    model_name = st.selectbox(
        "Select Model",
        ("RobCaamano/toxicity", "distilbert-base-uncased-finetuned-sst-2-english"),
    )
    submit = st.button("Submit", type="primary")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if submit:
    results = dict(d.values() for d in clf(text)[0])

    if model_name == "RobCaamano/toxicity":
        classes = {k: results[k] for k in results.keys() if not k == "toxic"}

        max_class = max(classes, key=classes.get)
        probability = classes[max_class]

        if results['toxic'] >= 0.5:
            result_df = pd.DataFrame({
                'Toxic': 'Yes',
                'Toxicity Class': [max_class],
                'Probability': [probability]
            })
        else:
            result_df = pd.DataFrame({
                'Toxic': 'No',
                'Toxicity Class': 'This text is not toxic',
            })

    elif model_name == "distilbert-base-uncased-finetuned-sst-2-english":
        result = max(results, key=results.get)
        probability = results[result]

        result_df = pd.DataFrame({
            'Result': [result],
            'Probability': [probability],
        })

    st.table(result_df)

    expander = st.expander("View Raw output")
    expander.write(results)
