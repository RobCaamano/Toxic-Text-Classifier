# Toxic Text Classifier

## Sections

- [About](#about)
- [Demo Site](#demo)

## About <a id="about"></a>

The Toxic Text Classifier is a user-friendly web application built with Streamlit that is designed to analyze and categorize text through toxicity classification and sentiment analysis. This project utilizes a fine-tuned BERT model for text classification, leveraging TensorFlow and Hugging Face Transformers for NLP capabilities.

### <ins> Classification Categories </ins>

| Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
| -------- | -------- | -------- | -------- | -------- | -------- |

### <ins> Available Models </ins>

The site includes various pre-trained models with different levels of training and weighting:

- **Toxicity - 1 Epoch**

- **Toxicity - 8 Epochs**

- **[Toxicity - Weighted](https://huggingface.co/RobCaamano/toxicity_weighted)**

- **[DistilBERT Base Uncased (SST-2)](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)**

## Hugging Face Space <a id="demo"></a>

This [Hugging Face Space](https://huggingface.co/spaces/RobCaamano/Finetuning_Language_Models-Toxic_Tweets) is best used within my web app. The models can also be downloaded from my [Hugging Face Profile](https://huggingface.co/RobCaamano).
