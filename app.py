import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)

st.title("Detecting Toxic Tweets")

demo = """Your words are like poison. They seep into my mind and make me feel worthless."""

text = st.text_area("Input Text", demo, height=250)

model_options = {
    "DistilBERT Base Uncased (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "Fine-tuned Toxicity Model": "RobCaamano/toxicity",
    "Fine-tuned Toxicity Model - Optimized": "RobCaamano/toxicity_optimized",
}
selected_model = st.selectbox("Select Model", options=list(model_options.keys()))

mod_name = model_options[selected_model]

tokenizer = AutoTokenizer.from_pretrained(mod_name)
model = AutoModelForSequenceClassification.from_pretrained(mod_name)

if selected_model in ["Fine-tuned Toxicity Model", "Fine-tuned Toxicity Model - Optimized"]:
    toxicity_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    model.config.id2label = {i: toxicity_classes[i] for i in range(model.config.num_labels)}

def get_toxicity_class(prediction):
    max_index = prediction.argmax()
    return model.config.id2label[max_index], prediction[max_index]

input = tokenizer(text, return_tensors="tf")
prediction = model(input)[0].numpy()[0]

if st.button("Submit", type="primary"):
    label, probability = get_toxicity_class(prediction)
    
    tweet_portion = text[:50] + "..." if len(text) > 50 else text

    if selected_model in ["Fine-tuned Toxicity Model", "Model 3.0"]:
        column_name = "Toxicity Class"
    else:
        column_name = "Prediction"

    if probability < 0.1:
        st.write("This text is not toxic.")

    df = pd.DataFrame(
        {
            "Text (portion)": [tweet_portion],
            column_name: [label],
            "Probability": [probability],
        }
    )

    st.table(df)
