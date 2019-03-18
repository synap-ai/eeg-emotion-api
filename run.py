# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

from keras.models import load_model
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

EEG_WINDOW_SIZE = 3564


def load():
    global model
    model = load_model('models/eeg_regression_model.h5')


def prepare_eeg(eeg):
    # TODO Add blink filter
    eeg = eeg[:EEG_WINDOW_SIZE]
    eeg = eeg.reshape((len(eeg), len(eeg[0]), len(eeg[0][0]), 1))

    return eeg


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.get_json("eeg"):
            eeg = np.array(flask.request.get_json["image"])
            eeg = prepare_eeg(eeg)

            pred = model.predict(eeg)
            data["predictions"] = [pred]
            data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load()
    app.run()
