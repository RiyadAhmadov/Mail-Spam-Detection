# Let's load libraries
import tensorflow as tf  # for tensorflow
import regex as re  # Regular expressions library
import numpy as np  # Numerical computing library
import seaborn as sns  # Statistical data visualization
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library
from sklearn.metrics import f1_score # Calculate f-score
from collections import Counter # word counter library
from sklearn.model_selection import train_test_split  # Model selection and evaluation
from tensorflow.keras.preprocessing.text import Tokenizer  # Text preprocessing for Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Sequence preprocessing for Keras
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
# Download WordNet if not already downloaded
nltk.download('wordnet')

import warnings as wg
wg.filterwarnings('ignore')
import pickle


# Let's import data
data = pd.read_csv('spam_ham_dataset.csv', index_col='Unnamed: 0')

# Let's do preprocessing in data
data['text'] = data['text'].apply(lambda x: str(x).replace('Subject: ', ''))
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))
del data['label_num']
# Let's divide data to two parts
ham_msg = data[data.label =='ham']
spam_msg = data[data.label =='spam']

# Let's randomly taking data from ham_msg (for imbalance problem solving)
ham_msg=ham_msg.sample(n=len(spam_msg),random_state=42)


balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)

# Let's get the English stopwords
stop_words = set(stopwords.words('english'))

# Function to clean text from stopwords
def remove_stopwords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Apply the remove_stopwords function to the 'text' column
balanced_data['cleaned_text'] = balanced_data['text'].apply(remove_stopwords)


# Let's initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()



# Function to perform lemmatization on text
def lemmatize_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize each token and remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the lemmatized tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

# Apply the lemmatize_text function to the 'cleaned_text' column
balanced_data['Cleaned_text_lemmatized'] = balanced_data['cleaned_text'].apply(lemmatize_text)


balanced_data = balanced_data[['label','Cleaned_text_lemmatized']]
balanced_data.columns = ['label','text']


# Let's create new column label with number
balanced_data['label_num']=balanced_data['label'].map({'ham':0,'spam':1})


# Let's divide dataset train and test data
train_msg, test_msg, train_labels, test_labels =train_test_split(balanced_data['text'],balanced_data['label_num'],test_size=0.2,random_state=434)

vocab_size= 900 # vocabulary size - only the 500 most frequent words will be kept in the vocabulary
oov_tok='<OOV>' # This variable defines the token to be used for out-of-vocabulary (OOV) words.
max_len=50 # This variable specifies the maximum length of sequences after padding or truncating.

#preprocessing making tokens out of text
token=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
token.fit_on_texts(train_msg)

# Let's look at index of words
word_index=token.word_index
word_index


padding_type='post'
truncate_type='post'
Trainning_seq=token.texts_to_sequences(train_msg)
Trainning_pad=pad_sequences(Trainning_seq,maxlen=50,padding=padding_type,truncating=truncate_type)


Testing_seq=token.texts_to_sequences(test_msg)
Testing_pad=pad_sequences(Testing_seq,maxlen=50,padding=padding_type,truncating=truncate_type)


# Let's create TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Let's calculate loss of model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'],optimizer='adam')


# Let's fit model for dataset
epoch = 20
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2) #  if val_loss is increase 3 time then epoc stop
history=model.fit(Trainning_pad, train_labels ,validation_data=(Testing_pad, test_labels),epochs=epoch,callbacks=[early_stop],verbose=2)


# Evaluate model
print(model.evaluate(Testing_pad, test_labels))


# Let's do test other ham value -  i get this mail from internet
predict_ham = ['I hope this email finds you well. I wanted to follow up on our conversation from yesterday regarding the project timeline. Please review the attached document and let me know if you have any questions or concerns. Looking forward to hearing from you. Best regards.']

# Let's predict new data
def predict_spam(predict_msg):
    new_seq = token.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen=50, padding=padding_type, truncating='post')
    prediction = model.predict(padded)
    rounded_prediction = np.round(prediction[0][0])
    return rounded_prediction

prediction = predict_spam(predict_ham)
if prediction == 1:
  a = 'spam'
  print(f'• Model: {a}')
else:
  a = 'ham'
  print(f'• Model: {a}')


# Save the model architecture as JSON
model_json = model.to_json()
with open("tensor_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights with the correct filename
model.save_weights("tensor_model_weights.weights.h5")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)