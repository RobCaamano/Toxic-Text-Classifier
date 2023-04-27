import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, pipeline
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)

st.title("Detecting Toxic Tweets")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

text = st.text_area("Input Text", demo, height=250)

model_options = {
    "DistilBERT Base Uncased (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "Fine-tuned Toxicity Model": "RobCaamano/toxicity",
}
selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

mod_name = model_options[selected_model]

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

if selected_model in ["Fine-tuned Toxicity Model"]:
    toxicity_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    model.config.id2label = {i: toxicity_classes[i] for i in range(model.config.num_labels)}

def get_most_toxic_class(predictions):
    return {model.config.id2label[i]: pred for i, pred in enumerate(predictions)}

input = tokenizer(text, return_tensors="tf")

if st.button("Submit", type="primary"):
    results = dict(d.values() for d in clf(text)[0])
    toxic_labels = get_most_toxic_class(results)

    tweet_portion = text[:50] + "..." if len(text) > 50 else text

    if len(toxic_labels) == 0:
        st.write("This text is not toxic.")
    else:
        max_toxic_class = max(toxic_labels, key=toxic_labels.get)
        max_probability = toxic_labels[max_toxic_class]

        df = pd.DataFrame(
            {
                "Text (portion)": [tweet_portion],
                "Toxicity Class": [max_toxic_class],
                "Probability": [max_probability],
            }
        )
        st.table(df)
