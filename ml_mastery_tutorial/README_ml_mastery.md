# Purpose

The purpose of this repo is to experiment with using machine-learning techniques to create useful metadata for [Opening-the-Archives] items that are metadata-deficient.

[Opening-the-Archives]: <https://repository.library.brown.edu/studio/collections/bdr:318399/>

---


# History

In the late-winter/early-spring of 2023 a few of us began meeting a few times per month to educate ourselves about machine-learning techniques that could potentially be useful in our digital-Library work. 

We identified a BDR task, where, for a particular collection of some 50,000 items -- some thousands of items have a blank metadata field. We wondered if we could use other item metadata fields to predict the missing metadata field -- using the tens of thousands of items where none of the metadata fields are missing. 

Initially we started with a very simple machine-learning classification tutorial. (add-link)

Then we went through another machine-learning tutorial. (add-link)

Then we took our data, and roughly followed that second tutorial, using copilot and chatgpt to see how far we could get.

In December, 2023, we re-assessed and pivoted. We had learned a lot, but had gotten stuck, in terms of working code. Part of this was because we only met for about an hour, when all three of us were available, which meant approximately three times a month, so some time was regularly spent getting up to speed with where we had left off.

Another significant factor was related to our extensive use of copilot and chatgpt to augment our limited real but limited knowledge. The problem with that was that this field is moving _fast_, and we'd regularly have code suggested that may well have been accurate for earlier versions of libraries, but was just off enough to confuse us -- so we spent lots of time figuring out syntax, without have quite enough of a solid foundation for how to evaluate whether we were on the right track conceptually.

In January 2024, we took a couple of sessions to find another tutorial that looked both simple and fit our goal, and [found one on multi-label classification]. We went through this tutorial, then came up with a very simple toy data-set that we thought we could convert to the type of data for the tutorial's model/approach. And we've just achieved success in getting the model to accurately predict the very simple missing toy-set data.

Our next step will be to take a small subset of our actual data and see if we can use this model.

In the future, if we're able to get this to work with our actual data, we'll then grapple with: 
- how to evaluate the accuracy of the machine-learning produced data
- how to make the decision whether to proceed with using it
- how we might approach being transparent that some of the metadata is ML-generated

[found one on multi-label classification]: <https://machinelearningmastery.com/multi-label-classification-with-deep-learning/>

---


# Usage

We'll likely refactor this repo, but at the moment...

```
$ cd /path/to/ml_bdr_ota_experiment/
$ source ../env/bin/activate
$ python ./ml_mastery_tutorial/main_script.py
```

Note that this assumes a virtual-environment has been populated from either `ml_bdr_ota_experiment/ml_mastery_tutorial/requirements_mac.txt` or `ml_bdr_ota_experiment/ml_mastery_tutorial/requirements.txt`

---

# Model

_chatgpt4 explanation_ 

The Python function `get_model()` creates and compiles a neural network model using Keras, a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It is designed to work with two parameters: `n_inputs` and `n_outputs`, which dictate the dimensionality of the input layer and the number of units in the output layer, respectively. Here's a step-by-step explanation of what happens inside this function:

1. Creating a Sequential Model:

	- `model = Sequential()`: This line initializes a new Sequential model. Sequential models are a linear stack of layers where you can just add layers via the .add() method. It's the simplest kind of Keras model for neural networks that are just composed of a single stack of layers connected sequentially.

2. Adding the First Dense Layer (Input Layer):

	- `model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))`: This line adds the first layer to the model, which is a densely-connected (fully connected) layer. The parameters specify the configuration of this layer:
		- 20 units: This is the number of neurons or nodes in the layer. Each unit will learn to represent a different aspect of the input data.
		- `input_dim=n_inputs`: This specifies the size of the input layer, i.e., the number of input features the model expects. For the first layer in the model, you need to explicitly state the size of the input layer.
		- `kernel_initializer='he_uniform'`: This initializer sets the initial weights of the layer according to a uniform distribution scaled by the number of input units, which is a good practice when using the ReLU activation function, as it helps in avoiding problems related to the initialization of weights.
		- `activation='relu'`: This applies the Rectified Linear Unit (ReLU) activation function to the output of the neurons in this layer. The ReLU function is commonly used due to its computational simplicity and because it allows models to learn faster and perform better.

3. Adding the Output Layer:

	- `model.add(Dense(n_outputs, activation='sigmoid'))`: This adds the output layer to the model. The number of units in the output layer is determined by n_outputs, which depends on the problem you're trying to solve (e.g., binary classification, multi-label classification).
	- `n_outputs units`: The number of neurons corresponds to the number of output classes or predictions the model should produce.
	- `activation='sigmoid'`: For binary classification or multi-label classification problems, the sigmoid activation function is used. It squashes the output of each neuron to a value between 0 and 1, which can be interpreted as a probability.

4. Compiling the Model:

	- `model.compile(loss='binary_crossentropy', optimizer='adam')`: This line configures the model for training.
	- `loss='binary_crossentropy'`: This specifies the loss function to use, which in this case is binary crossentropy, a common choice for binary classification problems. It measures the difference between the true labels and the predictions made by the model, aiming to minimize this difference during training.
	- `optimizer='adam'`: The Adam optimizer is specified here, which is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. Adam is popular due to its effectiveness in practice and minimal need for manual tuning of the learning rate.

After executing these steps, the function returns the compiled model, ready for training with input data and corresponding labels.

---


# Other

The [raw-data bdr-api query].

[raw-data bdr-api query]: <https://repository.library.brown.edu/api/search/?q=rel_is_member_of_collection_ssim:%22bdr:318399%22%20AND%20-rel_is_part_of_ssim:*%20AND%20ds_ids_ssim:MODS&rows=99999>

---

(end)
