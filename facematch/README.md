#facematch
A Facial Recognition Library

This library is built and modelled after [MM-DFR](https://arxiv.org/abs/1509.00244), which is a model of multiple CNNs which feed it's activations
into a stacked auto encoder for further dimensionality reduction. The output is a 512-length vector which can be seen
as the encoded representation of a given face.

##How to Use
Typically, you will instantiate the `API` object from `api.py` and call the `train(model_name)` method.
This method will persist the model to disk given a certain name that can be used to later retrieve the model.
To retrieve the encoded face representations you have to call `load_model(model_name)` which loads the models from disk.
Then you can call `get_face_vector(image)` as many times as you want.

There will be functions that will return a similarity score given some face vectors.
It is up to the user to determine they're threshold for what score determines which faces are the same.

If you want to train the model with your own dataset, you can call `add_image(user_id, image)` and it will be persisted to disk.
The only caveat is you have to retrain the model.
`remove_images()` removes all images that the user has added through `add_image`.

##Installation and Set Up
The `requirements.txt` show which python libs are required.
You'll need `keras 1.0.3`, `theano 0.8.1`, `numpy`, `scikit-image` and `dlib`.
Note: If you don't install the exact versions of `keras` and `theano` the library might break 
because models are persisted differently between versions.

The library assumes you're running on AWS g2.8xlarge. 
If you're not adjust the compile time constants in `consts.py` from 4 GPUs to the number of GPUs on your machine. 
Also we make certain assumptions about the name of the GPUs so this should also be adjusted. 
Note: There may be other things you need to adjust.

We use [NVIDIA's CUDA Deep Neural Network library](https://developer.nvidia.com/cudnn) (cuDNN) v3 and CUDA v7.
We use these to be able to run our code on the GPUs, without these our code only runs on the CPUs.
Make sure this is set up so our models train and score fast.

This library is memory intensive so one should setup swap space. 
The following [guide](https://www.digitalocean.com/community/tutorials/how-to-add-swap-on-ubuntu-14-04) is useful.
