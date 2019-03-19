from keras.models import load_model
import numpy as np
import tensorflow as tf
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config.update(
    DEBUG=True,
    MAX_CONTENT_LENGTH=50000000
)
model = None

EEG_WINDOW_SIZE = 3564

EEG_SENSORS = ['tp9', 'af7', 'af8', 'tp10']

def load():
    global model
    model = load_model('models/eeg_regression_model_2.h5')
    global graph
    graph = tf.get_default_graph()


def prepare_eeg(eeg):
    cleaned_eeg = []
    for sample in eeg:
        eeg_reading = []
        for k in EEG_SENSORS:
            eeg_reading.append(sample[k])
        cleaned_eeg.append(eeg_reading)
    cleaned_eeg = np.array([cleaned_eeg])
    # TODO Add blink filter, time stamps
    cleaned_eeg = cleaned_eeg.reshape((len(cleaned_eeg), len(cleaned_eeg[0]), len(cleaned_eeg[0][0]), 1))

    return cleaned_eeg


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.get_json("eeg"):
            eeg = prepare_eeg(flask.request.get_json("eeg")["eeg"][:EEG_WINDOW_SIZE])

            with graph.as_default():
                preds = model.predict(eeg)
                data["predictions"] = [preds.tolist()]
                data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load()
    app.run()
