Language Detection Model
This repository contains a Python implementation of a language detection model using Logistic Regression and TF-IDF vectorization. The model is trained to classify text into different languages based on character n-grams.

Table of Contents
Dataset
Dependencies
Preprocessing
Model Training
Evaluation
Usage
Results
Contributing
License
Dataset
The dataset used in this project is Language Detection.csv, which contains text samples labeled with their respective languages.

Dependencies
Make sure you have the following libraries installed:

pandas
numpy
scikit-learn
matplotlib
seaborn
You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn
Preprocessing
The preprocessing steps include:

Removing punctuation from the text.
Converting text to lowercase.
This is done using the remove_pun function:

python
Copy code
def remove_pun(text):
    for pun in string.punctuation:
        text = text.replace(pun, "")
    text = text.lower()
    return text
Model Training
The model uses TF-IDF vectorization on character n-grams (1 to 2 characters) and Logistic Regression for classification. The data is split into training and testing sets with a 75-25 split.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, pipeline, linear_model

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Vectorize the text
vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), analyzer='char')

# Create a pipeline
model_pipe = pipeline.Pipeline([('vec', vec), ('clf', linear_model.LogisticRegression())])

# Train the model
model_pipe.fit(x_train, y_train)
Evaluation
The model is evaluated using accuracy score and confusion matrix.

python
Copy code
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
Usage
To use the model for predicting the language of new text samples, you can use the predict method of the trained pipeline:

python
Copy code
model_pipe.predict(['the name is vincent'])
model_pipe.predict(['என் பெயர் பாலா'])
model_pipe.predict(['Goedemorgen!'])
model_pipe.predict(['बिल्कुल! यहाँ है एक सरल वाक्य हिंदी में'])
model_pipe.predict(['Bonjour, je m’appelle…'])
model_pipe.predict(['Коты любят играть с мячиками'])

Results
The model achieves high accuracy in detecting the languages of the text samples. The confusion matrix provides further insight into the model's performance across different languages.
