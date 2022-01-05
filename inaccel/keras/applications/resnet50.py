"""ResNet50 model for Keras.

# Reference
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

import h5py
import inaccel.coral as inaccel
import keras.utils.data_utils as data_utils
import keras.utils.generic_utils as generic_utils
import numpy as np
import os

from .imagenet_utils import decode_predictions
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

WEIGHTS_PATH = ('https://inaccel-demo.s3.amazonaws.com/models/'
                'resnet50_weights.h5')

class ResNet50:
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.

    # Arguments
        weights: one of 'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        classes: optional number of classes to classify images
            into, only to be specified if no `weights` argument is specified.

    # Raises
        ValueError: in case of invalid argument for `weights`.
    """
    def __init__(self, weights='imagenet', classes=1000):
        self.weights = weights
        self.classes = classes

        if not (weights in {'imagenet'} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`imagenet` (pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"`, '
                             '`classes` should be 1000')

        # Load weights.
        if weights == 'imagenet':
            weights_path = data_utils.get_file(
                'resnet50_weights.h5',
                WEIGHTS_PATH,
                cache_subdir='inaccel/models',
                md5_hash='9d0d81e932eaaff345d658f2a277372c')
            self.load_weights(weights_path)
        elif weights is not None:
            self.load_weights(weights)

    def load_weights(self, filepath):
        """Loads all layer weights from a HDF5 save file.

        # Arguments
            filepath: String, path to the weights file to load.
        """
        with h5py.File(filepath, mode='r') as f:
            if 'weights' in f:
                # Create model.
                with inaccel.allocator:
                    self.model = np.array(f['weights'][:])

    def predict(self, x,
                batch_size=None,
                steps=None,
                max_queue_size=10,
                workers=1):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: Input data. It could be:
                - A Numpy array (or array-like).
                - A generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample weights)`.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of generators, or
                `keras.utils.Sequence` instances (since they generate batches).
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case of mismatch between the provided
                input data and the model's expectations.
        """
        if batch_size is not None and data_utils.is_generator_or_sequence(x):
            raise ValueError('The `batch_size` argument must not be specified when'
                             ' using a generator or Sequence as an input.')

        if batch_size is None:
            # Backwards compatibility
            batch_size = 32

        # Case 1: generator-like. Input is Python generator, or Sequence object.
        if data_utils.is_generator_or_sequence(x):
            return self.predict_generator(
                x,
                steps=steps,
                max_queue_size=max_queue_size,
                workers=workers)

        # Case 2: Numpy array-like.
        outputs = []
        for start, stop in generic_utils.make_batches(len(x), batch_size):
            outputs.append(self.predict_on_batch(x[start:stop]))
        return np.vstack(outputs)

    def predict_generator(self, generator,
                          steps=None,
                          max_queue_size=10,
                          workers=1):
        """Generates predictions for the input samples from a data generator.

        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: Generator yielding batches of input samples
                or an instance of Sequence (keras.utils.Sequence)
                object.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: Maximum size for the generator queue.
            workers: Integer. Maximum number of processes to spin up
                when using process based threading.
                If unspecified, `workers` will default to 1.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        outputs = []
        use_sequence_api = isinstance(generator, data_utils.Sequence)
        if steps is None:
            if use_sequence_api:
                steps = len(generator)
            else:
                raise ValueError('`steps=None` is only valid for a generator'
                                 ' based on the `keras.utils.Sequence` class.'
                                 ' Please specify `steps` or use the'
                                 ' `keras.utils.Sequence` class.')

        queue = Queue(max_queue_size)
        thread = thread = Thread(target = self._waiter_, args = (queue, outputs))
        thread.start()

        try:
            batches = ThreadPoolExecutor(workers).map(lambda _: next(generator), range(0, steps))

            for batch in batches:
                if isinstance(batch, tuple):
                    if len(batch) == 2:
                        x, _ = batch
                    elif len(batch) == 3:
                        x, _, _ = batch
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(batch))
                else:
                    # Assumes a generator that only
                    # yields inputs (not targets and sample weights).
                    x = batch

                queue.put(self._submit_(x))
        finally:
            queue.put(None)
            thread.join()

        return np.vstack(outputs)

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        # Arguments
            x: Input samples, as a Numpy array.

        # Returns
            Numpy array(s) of predictions.
        """
        return self._wait_(self._submit_(x))

    def _submit_(self, input):
        if not inaccel.allocator.handles(input) or input.dtype != np.int8:
            with inaccel.allocator:
                input = np.array(input, dtype = np.int8)

        n = np.uint32(len(input))

        with inaccel.allocator:
            output = np.ndarray((n, 5), dtype = np.uint16)

        resnet50 = inaccel.request("xilinx.com.researchlabs.resnet50")
        resnet50.arg(input).arg(output).arg(self.model).arg(n)

        return {
            'in': input,
            'out': output,
            'weightsmem': self.model,
            'numReps': n,
            '_': inaccel.submit(resnet50)
        }

    def _wait_(self, accelerator):
        accelerator['_'].result()

        return accelerator['out']

    def _waiter_(self, queue, outputs):
        accelerator = queue.get()
        while accelerator is not None:
            outputs.append(self._wait_(accelerator))
            accelerator = queue.get()
