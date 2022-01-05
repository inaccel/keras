"""MobileNet model for Keras.

# Reference
  - [MobileNets: Efficient Convolutional Neural Networks
     for Mobile Vision Applications](
      https://arxiv.org/abs/1704.04861)
"""

import inaccel.coral as inaccel
import keras.utils.data_utils as data_utils
import keras.utils.generic_utils as generic_utils
import numpy as np

from .imagenet_utils import decode_predictions
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

class MobileNet:
    """Instantiates the MobileNet architecture."""

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

        mobilenet = inaccel.request("xilinx.com.researchlabs.mobilenet")
        mobilenet.arg(input).arg(output).arg(n)

        return {
            'in': input,
            'out': output,
            'numReps': n,
            '_': inaccel.submit(mobilenet)
        }

    def _wait_(self, accelerator):
        accelerator['_'].result()

        return accelerator['out']

    def _waiter_(self, queue, outputs):
        accelerator = queue.get()
        while accelerator is not None:
            outputs.append(self._wait_(accelerator))
            accelerator = queue.get()
