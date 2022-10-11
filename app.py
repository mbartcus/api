###to run the Flask server
# FLASK_APP=Bartcus_Marius_API_092022.py flask run
###
import os
from flask import Flask, request, render_template, jsonify
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd


app = Flask(__name__)


@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    results_data_path = os.path.join("results")
    model_name = "lstm_glove_embedded"
    model_file_path = os.path.join(results_data_path, model_name)

    global model, data
    model = load_model(model_file_path)
    data = pd.read_pickle("processed_nlp_data.pkl.gz")

    print('[INFO] : Model and preprocessed Glove data loaded')


def predict(my_tweet):
    # Prediction:
    v_size = 100
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data.CleanText)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    X = pad_sequences(tokenizer.texts_to_sequences(text),
                            maxlen = v_size)

    y_test_pred_proba = model.predict(
        X,
        batch_size=128
    ) # the probability of positive or negative

    class_pos_neg = [round(pred_proba[0]) for pred_proba in y_test_pred_proba] # if 1 the tweet is positive if 0 the tweet is negative

    return class_pos_neg[0], y_test_pred_proba[0,0]

# API
@app.route("/api/")
def sentiment_tweet():
    #dictionnaire = {
    #        'type': 'Prévision de température',
    #    'valeurs': [24, 24, 25, 26, 27, 28],
    #    'unite': "degrés Celcius"
    #}
    #return jsonify(dictionnaire)

    my_tweet = request.args.get("my_tweet")
    if not my_tweet:
        my_tweet = ''
        pos = 0.50
        neg = 0.50
    else:
        class_pos_neg, y_test_pred_proba = predict(my_tweet)


    if class_pos_neg==0:
        sentiment="Negative"
    else:
        sentiment="Positive"
    dictionnaire = {
        'sentiment': sentiment,
        'prob': str(y_test_pred_proba),
    }

    return jsonify(dictionnaire)


# API TEST
@app.route("/test/")
def test_api():
    dictionnaire = {
        'type': 'Prévision de température',
        'valeurs': [24, 24, 25, 26, 27, 28],
        'unite': "degrés Celcius"
    }
    return jsonify(dictionnaire)

if __name__ == "__main__":
    app.run(debug==True)
