---
title: Finetuning Language Models-Toxic Tweets
emoji: 🌖
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

# Finetuning_Language_Models-Toxic_Tweets

# App Landing Page
[link](https://sites.google.com/view/detecting-toxicity-in-text/home)

# Documentation
### train_weighted.py
This code is for training a multi-label text classification model to detect different types of toxic comments, using the TensorFlow library along with the Hugging Face Transformers library. It first imports the required libraries and sets up GPU support if available.

1. Import the necessary libraries and modules.
2. Set the visible GPU device for TensorFlow to use.
3. Check if a GPU is found and print the appropriate message.
4. Define the base model, output directory, checkpoint path, and batch size for training.
5. Define the toxic comment labels and create dictionaries to map between labels and their corresponding IDs.
6. Load the dataset from a CSV file using the Hugging Face 'datasets' library.
7. Load the tokenizer using the base model.
8. Define the 'process_data' function to preprocess each row of the dataset by encoding the text and creating a list of labels.
9. Load the pre-trained model with the specified base model, problem type, number of labels, and label-to-ID mappings.
10. Preprocess the dataset using the 'process_data' function and remove unnecessary columns.
11. Prepare the TensorFlow dataset for training with appropriate shuffling and tokenization.
12. Set up the callbacks for checkpointing, pushing to the Hugging Face hub, reducing learning rate on plateau, and early stopping.
13. Compile the model with the Adam optimizer, binary cross-entropy loss, and precision and recall metrics.
14. Train the model using the preprocessed dataset and specified callbacks.

The final model will be trained to classify comments into six categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The trained model will be saved to the specified output directory, with checkpoints saved to the checkpoint path. The model will also be pushed to the Hugging Face hub for easy sharing and deployment.

### app.py
This code creates a web application using the Streamlit library for classifying text into different toxicity categories or sentiment analysis, depending on the selected model. It utilizes the Hugging Face Transformers library for tokenization and model inference.

1. Import the necessary libraries and modules.
2. Set up a Streamlit title for the application.
3. Define demo options for text input.
4. Create a selection box to choose a demo example, and a text area to input or modify text.
5. Set up a submit button and define a mapping between model display names and their Hugging Face model IDs.
6. Create a container to select the model from a dropdown list and a submit button.
7. Load the tokenizer and model based on the selected model.
8. Set up a classification pipeline with the loaded model and tokenizer.
9. Tokenize the input text.
10. When the submit button is clicked, process the input text and obtain classification results.
11. If a toxicity model is used, create a DataFrame with the toxicity information.
12. If a sentiment analysis model is used, create a DataFrame with the sentiment information.
13. Display the DataFrame containing the results in a table.
14. Provide an option to view the raw output of the classification.

The web application allows users to input text, select a pre-trained model (toxicity or sentiment analysis), and receive the classification results. The application supports three models: a toxicity model trained for 1 epoch, a toxicity model trained for 8 epochs, and a sentiment analysis model (DistilBERT Base Uncased) fine-tuned on the SST-2 dataset.
