# use final model to generate captions on a new photo

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep002-loss3.862-val_loss3.890.h5') # black dog is swimming in water

# model stopped improving at the 4th epoch - stopped after completing 10th epoch
#initial model:'model-ep001-loss4.505-val_loss4.057.h5'-black dog is running through the water
#3rd best model:'model-ep002-loss3.862-val_loss3.890.h5'-black dog is running through the water
#2nd best model: 'model-ep003-loss3.677-val_loss3.855.h5'-black dog is running through the water
#best model: 'model-ep004-loss3.588-val_loss3.825.h5'-black dog is swimming in water

# load and prepare the photographs
#photo = extract_features('example.jpg')
photo1 = extract_features('TF_Test/1.png') # startseq man in black shirt is standing on the street endseq
photo2 = extract_features('TF_Test/2.png')
photo3 = extract_features('TF_Test/3.png')
photo4 = extract_features('TF_Test/4.png')
photo5 = extract_features('TF_Test/5.png')
photo6 = extract_features('TF_Test/6.png')
photo7 = extract_features('TF_Test/7.png')
photo8 = extract_features('TF_Test/8.png')
photo9 = extract_features('TF_Test/9.png')
photo10 = extract_features('TF_Test/10.png')
photo11 = extract_features('TF_Test/11.png')
photo12 = extract_features('TF_Test/12.png')
photo13 = extract_features('TF_Test/13.png')
photo14 = extract_features('TF_Test/14.png')

# generate description
#description = generate_desc(model, tokenizer, photo, max_length)
#print(description)

description1 = generate_desc(model, tokenizer, photo1, max_length)
print('1.png caption: ' + description1)
description2 = generate_desc(model, tokenizer, photo2, max_length)
print('2.png caption: ' + description2)
description3 = generate_desc(model, tokenizer, photo3, max_length)
print('3.png caption: ' + description3)
description4 = generate_desc(model, tokenizer, photo4, max_length)
print('4.png caption: ' + description4)
description5 = generate_desc(model, tokenizer, photo5, max_length)
print('5.png caption: ' + description5)
description6 = generate_desc(model, tokenizer, photo6, max_length)
print('6.png caption: ' + description6)
description7 = generate_desc(model, tokenizer, photo7, max_length)
print('7.png caption: ' + description7)
description8 = generate_desc(model, tokenizer, photo8, max_length)
print('8.png caption: ' + description8)
description9 = generate_desc(model, tokenizer, photo9, max_length)
print('9.png caption: ' + description9)
description10 = generate_desc(model, tokenizer, photo10, max_length)
print('10.png caption: ' + description10)
description11 = generate_desc(model, tokenizer, photo11, max_length)
print('11.png caption: ' + description11)
description12 = generate_desc(model, tokenizer, photo12, max_length)
print('12.png caption: ' + description12)
description13 = generate_desc(model, tokenizer, photo13, max_length)
print('13.png caption: ' + description13)
description14 = generate_desc(model, tokenizer, photo14, max_length)
print('14.png caption: ' + description14)