# Toxic Text Classifier

## Sections

- [About](#about)
- [Example Output](#example)
- [How to Use](#usage)
- [Hugging Face](#space)

## About <a id="about"></a>

The Toxic Text Classifier is a user-friendly web application built with Streamlit that is designed to analyze and categorize text through toxicity classification and sentiment analysis. This project utilizes a fine-tuned BERT model for text classification, leveraging TensorFlow and Hugging Face Transformers for NLP capabilities.

### <ins> Classification Categories </ins>

| Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
| -------- | -------- | -------- | -------- | -------- | -------- |

### <ins> Available Models </ins>

The site includes various pre-trained models with different levels of training and weighting. This shows my development progress and the implementation of class weights for higher accuracy:

- **Toxicity - 1 Epoch**

- **Toxicity - 8 Epochs**

- **[Toxicity - Weighted](https://huggingface.co/RobCaamano/toxicity_weighted)**

- **[DistilBERT Base Uncased (SST-2)](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)**

## Example Output <a id="example"></a>

<img src="https://github.com/user-attachments/assets/61abe86c-306f-4d21-95dd-8f81b75053a0" alt="Output" width="400" height=600/>


## How to use: <a id="usage"></a>

1. Navigate to the [web app homepage](https://sites.google.com/view/detecting-toxicity-in-text/home)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Note:** It may take a moment for the streamlit app to load

3. Select a demo to view an example of a certain type of toxicity in action

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**OR**

4. Type the text you want to analyze for toxicity

5. Select a model for either toxicity classification or sentiment analysis

6. Hit the submit button to view the output

## Hugging Face <a id="space"></a>

This [Hugging Face Space](https://huggingface.co/spaces/RobCaamano/Finetuning_Language_Models-Toxic_Tweets) is best used within my web app. The models can also be downloaded from my [Hugging Face Profile](https://huggingface.co/RobCaamano).
