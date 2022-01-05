"""Utilities for ImageNet data preprocessing & prediction decoding."""

import json
import keras.utils.data_utils as data_utils

CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/'
                    'data/imagenet_class_index.json')

def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy array encoding a batch of predictions.
        top: Integer, how many top-guesses to return. Defaults to 5.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: In case of invalid shape of the `preds` array
            (must be 2D).
    """
    global CLASS_INDEX

    if len(preds.shape) != 2 or preds.shape[1] != 5:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 5)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = data_utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash='c2c37ea517e94d9795004a39431a14cb')
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred[:min(top, 5)]
        result = [tuple(CLASS_INDEX[str(i)]) for i in top_indices]
        results.append(result)
    return results
