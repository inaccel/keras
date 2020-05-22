import numpy as np
import time

from inaccel.keras.applications.resnet50 import decode_predictions, ResNet50
from inaccel.keras.preprocessing.image import ImageDataGenerator, load_img

model = ResNet50(weights='imagenet')

data = ImageDataGenerator(dtype='int8')
images = data.flow_from_directory('imagenet/', target_size=(224, 224), class_mode=None, batch_size=64)

begin = time.monotonic()
preds = model.predict(images, workers=16)
end = time.monotonic()

print('Duration for', len(preds), 'images: %.3f sec' % (end - begin))
print('FPS: %.3f' % (len(preds) / (end - begin)))

dog = load_img('data/dog.jpg', target_size=(224, 224))
dog = np.expand_dims(dog, axis=0)

elephant = load_img('data/elephant.jpg', target_size=(224, 224))
elephant = np.expand_dims(elephant, axis=0)

images = np.vstack([dog, elephant])

preds = model.predict(images)

print('Predicted:', decode_predictions(preds, top=1))
