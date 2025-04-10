# Spam-Email-Detection
This repository contains the implementation of a binary classification model designed to distinguish between spam (label: 1) and non-spam (label: 0) emails using a feedforward neural network.\
The project demonstrates how traditional Natural Language Processing (NLP) techniques can be integrated with neural networks to build robust and scalable spam detection systems.\
Problem Statement:\
Design and implement a binary classifier to detect whether an email is spam or legitimate (ham). The input dataset consists of raw email texts and their corresponding binary labels.\
*Methodology:*
1. Data Preprocessing\
•	The dataset is ingested from a .csv file containing email texts and their labels.\
•	Emails are cleaned by converting to lowercase and removing non-alphabetic characters using regular expressions.\
•	The word "Subject" is explicitly removed from the email content to reduce noise.
2. Feature Extraction\
•	Cleaned text data is converted into numeric vectors using TF-IDF (Term Frequency-Inverse Document Frequency) representation.\
•	A vocabulary size of 1,000 words is used to capture important textual patterns while managing dimensionality.
3. Train-Test Split\
•	Data is split into 80% training and 20% testing using scikit-learn’s train_test_split().
4. Model Architecture\
•	A feedforward neural network is implemented using TensorFlow/Keras.\
•	Architecture Details:\
o	Input Layer: 1,000-dimensional TF-IDF features\
o	Hidden Layer 1: 128 neurons, ReLU activation, followed by a 50% dropout\
o	Hidden Layer 2: 64 neurons, ReLU activation, followed by a 30% dropout\
o	Output Layer: 1 neuron with sigmoid activation for binary classification
5. Training Configuration\
•	The model is compiled with:\
o	Loss Function: Binary Crossentropy\
o	Optimizer: Adam (learning rate = 0.001)\
o	Metrics: Accuracy\
•	The network is trained for 10 epochs with a batch size of 32, using 20% of the training set as a validation set.
6. Evaluation\
•	After training, the model’s predictions are evaluated on the test set using:\
o	Accuracy Score\
o	Precision, Recall, F1-Score via classification_report from scikit-learn\

*Results:*\
The model demonstrates solid performance in classifying spam vs. non-spam emails based on textual patterns. The performance metrics and training logs provide insights into the convergence and generalization capability of the network.
 
