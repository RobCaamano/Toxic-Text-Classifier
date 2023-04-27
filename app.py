import streamlit as st
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

with col2:
    st.subheader("Classification")

with col3:
    st.subheader("Probability")


input = tokenizer(text, return_tensors="tf")

if submit:
    results = dict(d.values() for d in clf(text)[0])
    classes = {k: results[k] for k in results.keys() if not k == "toxic"}

    max_class = max(classes, key=classes.get)

    with col2:
        st.write(f"#### {max_class}")

    with col3:
        st.write(f"#### **{classes[max_class]:.2f}%**")

    if results["toxic"] < 0.5:
        st.success("This tweet is unlikely to be be toxic!", icon=":white_check_mark:")
    else:
        st.warning('This tweet is likely to be toxic.', icon=":warning:")
    
    expander = st.expander("Raw output")
    expander.write(results)
