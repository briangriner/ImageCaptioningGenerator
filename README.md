# Image Captioning Generator
Neural Image Captioning Generator with Attention

**Description**: Captioning generator is based on the book by Jason Brownlee, PhD. [Deep Learning for NLP](https://machinelearningmastery.com/deep-learning-for-nlp/) and his post [How to Develop a Deep Learning Photo Captioning Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/).  

**Data**: _Flikr8k_ can be requested from the [Department of Computer Science, University of Illinois at Urbana-Campaign](https://forms.illinois.edu/sec/1713398)

**Dependencies**: Scripts for the generator were created in a Python 3.6 conda environment using the intel distribution of python (see environment.yml - Not all libraries in this file are required to run the generator. The primary packages used are keras and TensorFlow and the Python libraries for images and strings. I recommend using Anaconda3 to create a conda environment for the application. The environment.yml file can be imported to manage dependencies).

**Memory requirements**: This version of the generator was run on a laptop with 16GB of RAM. Using the [Intel Distribution of Python](https://software.intel.com/en-us/distribution-for-python) greatly reduces the training time on a laptop with Intel CPUs. 

**Neural Image Captioning Architecture**: CNN + LSTM w/ attention

![](https://raw.githubusercontent.com/briangriner/ImageCaptioningGenerator/master/model.png)
