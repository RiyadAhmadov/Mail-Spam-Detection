import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import numpy as np  
import warnings
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import regex as re

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create a FastAPI app instance
app = FastAPI()

# Load the model architecture from JSON file
with open("tensor_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("tensor_model_weights.weights.h5")

# Compile the loaded model (if needed)
loaded_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['accuracy'], optimizer='adam')

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    # Join the filtered tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# HTML content for the prediction form
html_content = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Mail Detection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-image: url('https://assets-global.website-files.com/60fb23dfe1da5e54ffb7846f/65de4a17d82c28383166ec69_Solvingtheinboundspam.jpeg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }

        .container {
            max-width: 500px;
            width: 100%;
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.8); /* Adding opacity to make it semi-transparent */
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #333; /* Adjusting font color */
        }

        input[type="text"],
        input[type="email"] {
            width: 95%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        button {
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        .result {
            text-align: center;
            font-weight: bold;
            font-size: 20px; /* Increasing font size */
            color: #333; /* Adjusting font color */
        }

        #logo {
            display: block;
            margin: 0 auto;
            width: 100px;
            height: 100px;
            margin-bottom: 20px;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0, 0, 0, 0.4);
        }

        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            border-radius: 10px;
            animation: modal-show 0.3s ease-out; /* Adding animation */
        }

        @keyframes modal-show {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
        }

        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
            cursor: pointer;
        }

        .logo-container {
            text-align: center;
            margin-bottom: 20px;
            font-size: 30px; /* Adjust the font size as needed */
        }

        .modal-logo {
            width: 100px;
            height: 100px;
        }
    </style>
</head>

<body>
    <div class="container">
        <img id="logo" src="https://cdn-icons-png.flaticon.com/512/10733/10733107.png" alt="Logo">
        <h1>Spam Mail Detection</h1>
        <input type="text" id="nameInput" placeholder="Your Name...">
        <input type="email" id="mailInput" placeholder="Enter mail content...">
        <button onclick="predict()">Predict</button>
        <div id="myModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <div class="logo-container">
                    <img class="modal-logo" src="https://cdn-icons-png.flaticon.com/512/10733/10733107.png" alt="Logo">
                </div>
                <p class="result" id="predictionResult"></p>
            </div>
        </div>
    </div>
    <script>
        async function predict() {
            const name = document.getElementById('nameInput').value;
            const mail = document.getElementById('mailInput').value;
            const response = await fetch(`/predict/?name=${name}&mail=${mail}`);
            const data = await response.json();
            const predictionResult = document.getElementById('predictionResult');
            predictionResult.textContent = `${name}, your mail is ${data.prediction}`;
            // Show the modal
            const modal = document.getElementById("myModal");
            modal.style.display = "block";
        }

        // Close the modal when the user clicks on the close button
        const closeBtn = document.getElementsByClassName("close")[0];
        closeBtn.onclick = function () {
            const modal = document.getElementById("myModal");
            modal.style.display = "none";
        }

        // Close the modal when the user clicks anywhere outside of the modal
        window.onclick = function (event) {
            const modal = document.getElementById("myModal");
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    </script>
</body>

</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)


@app.get("/predict/")
async def predict(mail: str = Query(..., title="Mail Content")):

    # Preprocess the input text
    preprocessed_mail = preprocess_text(mail)
    
    # Tokenize and pad the preprocessed text
    sequence = tokenizer.texts_to_sequences([preprocessed_mail])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(padded_sequence)
    
    # Convert prediction result to binary (0 or 1)
    prediction = int(np.round(prediction[0][0]))
    if prediction == 1:
        a = 'spam'  
        print(f'• Model: {a}')
    else:
        a = 'ham'
    print(f'• Model: {a}')

    # Return the prediction result
    return {"prediction": a}

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info",
    )