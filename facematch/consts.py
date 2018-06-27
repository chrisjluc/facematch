import os

# Model Names
NN1 = 'nn1'
NN2 = 'nn2'
cnn_h1 = 'CNNH1'
cnn_p1 = 'CNNP1'
cnn_p2 = 'CNNP2'
cnn_p3 = 'CNNP3'
cnn_p4 = 'CNNP4'
cnn_p5 = 'CNNP5'
cnn_p6 = 'CNNP6'
sae_p1 = 'SAEP1'
sae_p2 = 'SAEP2'
sae_p3 = 'SAEP3'

# Model CNN consts
cnn_batch_size = 32
cnn_validation_split = .15
cnn_nb_epoch = 10
cnn_activation_batch_size = 256

# Model SAE consts
sae_batch_size = 256
sae_nb_epoch = 50
sae_validation_split =.15
sae_p1_input_size = 3584
sae_p1_encoding_size = 2048
sae_p2_encoding_size = 1024
sae_p3_encoding_size = 512
sae_regularizer = 10e-5

# Image processing
norm_shape = 230, 230
nn1_input_shape = 1, 100, 100
nn2_input_shape = 1, 165, 120
noise_width = 15

# Storage - Paths
data_path = '/home/ubuntu/FRS/facematch/.data'
face_predictor_path = os.path.join(
        data_path, 'shape_predictor_68_face_landmarks.dat')
image_path = os.path.join(data_path, 'images')
model_path = os.path.join(data_path, 'models')
test_data_path = '/home/ubuntu/FRS/facematch/tests/.data'
test_image_path = os.path.join(test_data_path, 'lfw')

# Storage - Extensions
image_ext = '_image.npy'
landmarks_ext = '_landmarks.npy'
features_ext = '_features.npy'
activations_ext = '_activations.npy'
encoder_ext = '_sae_encoder.npy'
autoencoder_ext = '_sae_autoencoder.npy'
json_ext = '.json'
h5_ext = '.h5'
jpg_ext = '.jpg'

# Tasks
num_gpus = 4
