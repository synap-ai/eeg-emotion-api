import numpy as np
import flask
import io
import pickle
import math
from collections import defaultdict

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config.update(
    DEBUG=True,
    MAX_CONTENT_LENGTH=50000000
)
arousal_model = None
pleasure_model = None

EEG_SENSORS = ['af7', 'af8']

sf = 256
window = 1.00
alpha = [8, 12]
beta = [12, 30]

def load():
    global arousal_model
    arousal_model = pickle.load(open('models/eeg_arousal_classify_model.pickle', 'rb'))
    global pleasure_model
    pleasure_model = pickle.load(open('models/eeg_pleasure_classify_model.pickle', 'rb'))


def prepare_eeg(eeg):
    # TODO Add blink filter
    eeg_readings = defaultdict(list)
    for sample in eeg:
        for k in EEG_SENSORS:
            if math.isnan(sample[k]):
                sample[k] = 0
            eeg_readings[k].append(sample[k])
    
    band_powers = []
    for i in range(0, len(eeg_readings['af7']) - sf, 16):
        af7 = eeg_readings['af7'][i:i+sf]
        af8 = eeg_readings['af8'][i:i+sf]

        af7_alpha = bandpower(af7, sf, alpha, window)
        af7_beta = bandpower(af7, sf, beta, window)
        af8_alpha = bandpower(af8, sf, alpha, window)
        af8_beta = bandpower(af8, sf, beta, window)
        band_powers.append([af7_alpha, af7_beta, af8_alpha, af8_beta])

    return np.array(band_powers)


@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.get_json('eeg'):
            eeg_data = flask.request.get_json('eeg')['eeg']
            eeg_data = prepare_eeg(eeg_data)
            arousal_preds = arousal_model.predict(eeg_data)
            pleasure_preds = pleasure_model.predict(eeg_data)
            data['predictions'] = {
                'arousal': [arousal_preds.tolist()],
                'pleasure': [arousal_preds.tolist()],
            }
            data['success'] = True

    return flask.jsonify(data)

def bandpower(data, sf, band, window_sec=None, relative=False):
    '''Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    '''
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

if __name__ == '__main__':
    print(('* Loading model and Flask starting server...'
           'please wait until server has fully started'))
    load()
    app.run()
