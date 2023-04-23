import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)

st.title("Detecting Toxic Tweets")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

text = st.text_area("Input text", demo, height=250)

model_options = {
    "DistilBERT Base Uncased (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "Fine-tuned Toxicity Model": "RobCaamano/toxicity_distilbert",
}
selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

mod_name = model_options[selected_model]

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)

if selected_model == "Fine-tuned Toxicity Model":
    toxicity_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    model.config.id2label = {i: toxicity_classes[i] for i in range(model.config.num_labels)}

def get_highest_toxicity_class(prediction):
    max_index = prediction.argmax()
    max_value = prediction[max_index]
    return [(model.config.id2label[i], pred_value) for i, pred_value in enumerate(prediction) if pred_value >= 0.5]

input = tokenizer(text, return_tensors="tf")
prediction = model(input)[0].numpy()[0]

if st.button("Submit", type="primary"):
    labels_with_probabilities = get_highest_toxicity_class(prediction)
    
    tweet_portion = text[:50] + "..." if len(text) > 50 else text

    if selected_model == "Fine-tuned Toxicity Model":
        column_name = "Toxicity Classes"
    else:
        column_name = "Prediction"

    if not labels_with_probabilities:
        st.write("This tweet is not toxic.")
    else:
        df = pd.DataFrame(
            {
                "Tweet (portion)": [tweet_portion] * len(labels_with_probabilities),
                column_name: [label for label, _ in labels_with_probabilities],
                "Probability": [probability for _, probability in labels_with_probabilities],
            }
        )
        st.table(df)
