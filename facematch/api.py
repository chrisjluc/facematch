import consts
import gc

import image_processing as ip
import data_processing as dp
import numpy as np

from image import Image
from storage import Reader, Writer
from models import NN1Model, NN2Model, AutoEncoderModel
from tasks import TrainingTask, ActivationExtractionTask, TrainingAutoEncoderTask
from task_manager import TaskManager

class API(object):

    def __init__(self):
        self.reader = Reader()
        self.writer = Writer()

    def load_model(self, name=''):
        """
        Retrieves model if it has been trained

        Note: After this function is called, we won't be able to
        call train() because the theano context exists in this process.
        GPUs will have initialization error
        """
        self.models = [
                NN2Model(name + consts.cnn_h1),
                NN1Model(name + consts.cnn_p1),
                NN1Model(name + consts.cnn_p2),
                NN1Model(name + consts.cnn_p3),
                NN1Model(name + consts.cnn_p4),
                NN1Model(name + consts.cnn_p5),
                NN1Model(name + consts.cnn_p6),
                AutoEncoderModel(name + consts.sae_p1),
                AutoEncoderModel(name + consts.sae_p2),
                AutoEncoderModel(name + consts.sae_p3),
                ]

        for model in self.models:
            model.load()

    def get_face_vector(self, image):
        """
        Given an image convert it to a vector.
        This vector is a representation of the face.
        """
        im = Image(image)
        data = dp.create_training_data_for_mmdfr([[im]])
        cnn_models = self.models[:7]
        sae_models = self.models[7:]
        activations = np.concatenate(
                tuple(model.get_activations(_data) for model, _data in zip(cnn_models, data)),
                axis=1)
        for model in sae_models:
            activations = model.get_activations(activations)
        return activations

    def train(self,
            name='',
            use_test_data=True,
            train_cnns=True,
            extract_activations=True,
            train_sae=True
        ):
        """
        Params:
        name: The name of the model that the user refers to
            this allows for multiple models to be persisted.
        use_test_data: On top of the images the user has added
            train on LFW database.

        Trains the mmdfr model described in this paper:
        https://arxiv.org/abs/1509.00244

        1) Images are extracted from persisted storage
        2) Data is augmented and transformed for the CNNs
        3) Since we have multiple GPUs we create tasks and run
            them on each of the GPUs to maximize utilization.
            The tasks involve having the keras models which run efficiently
            on the GPUs, so that's why we offload them to the GPUs.
        4) Training CNNs, extracting activations from these CNNS and use it
            to train the SAE.

        All intermediate data is persisted along with the models so it could
        be retrieved later if necessary.
        """
        cnn_h1 = name + consts.cnn_h1
        cnn_p1 = name + consts.cnn_p1
        cnn_p2 = name + consts.cnn_p2
        cnn_p3 = name + consts.cnn_p3
        cnn_p4 = name + consts.cnn_p4
        cnn_p5 = name + consts.cnn_p5
        cnn_p6 = name + consts.cnn_p6
        sae_p1 = name + consts.sae_p1
        sae_p2 = name + consts.sae_p2
        sae_p3 = name + consts.sae_p3

        needs_data = train_cnns or extract_activations

        if needs_data:
            # Get images the user has added
            user_ids = self.reader.get_user_ids(consts.image_path)
            images = []
            for user_id in user_ids:
                images.append(self.reader.get_images(user_id, consts.image_path))

        # Get test images from LFW to augment the model's training data
        if needs_data and use_test_data:
            test_user_ids = self.reader.get_user_ids_by_descending_jpg_counts(
                    consts.test_image_path, 600)
            test_images = []
            for user_id in test_user_ids:
                test_images.append(self.reader.get_images(user_id, consts.test_image_path))
            user_ids += test_user_ids
            images += test_images

            # Attempt to free unused memory
            del test_images, test_user_ids
            gc.collect()

        if needs_data:
            # Data augmentation
            images = self._augment_data(images)
            # Create proper image windows and sizes for each CNN
            data = dp.create_training_data_for_mmdfr(images)

            # Attempt to free unused memory
            del images
            gc.collect()

        if train_cnns:
            # Training CNNs
            tasks = [
                TrainingTask(NN2Model, data[0], data[7], user_ids, cnn_h1),
                TrainingTask(NN1Model, data[1], data[7], user_ids, cnn_p1),
                TrainingTask(NN1Model, data[2], data[7], user_ids, cnn_p2),
                TrainingTask(NN1Model, data[3], data[7], user_ids, cnn_p3),
                TrainingTask(NN1Model, data[4], data[7], user_ids, cnn_p4),
                TrainingTask(NN1Model, data[5], data[7], user_ids, cnn_p5),
                TrainingTask(NN1Model, data[6], data[7], user_ids, cnn_p6)
                ]
            task_manager = TaskManager(tasks)
            task_manager.run_tasks()

        if extract_activations:
            # Extracting activations from 2nd last layer for SAE
            tasks = [
                ActivationExtractionTask(NN2Model, cnn_h1, data[0], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p1, data[1], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p2, data[2], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p3, data[3], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p4, data[4], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p5, data[5], user_ids),
                ActivationExtractionTask(NN1Model, cnn_p6, data[6], user_ids)
                ]
            task_manager = TaskManager(tasks)
            task_manager.run_tasks()

        if needs_data:
            # These references aren't used anymore
            # This should release a lot of used memory here
            del data, user_ids
            gc.collect()

        if train_sae:
            # Training first layer in SAE
            activations = np.concatenate((
                self.reader.load_activations(cnn_h1),
                self.reader.load_activations(cnn_p1),
                self.reader.load_activations(cnn_p2),
                self.reader.load_activations(cnn_p3),
                self.reader.load_activations(cnn_p4),
                self.reader.load_activations(cnn_p5),
                self.reader.load_activations(cnn_p6)
                ), axis=1)
            tasks = [TrainingAutoEncoderTask(sae_p1, activations, consts.sae_p1_input_size, consts.sae_p1_encoding_size)]
            task_manager = TaskManager(tasks)
            task_manager.run_tasks()

            # Training second layer in SAE
            activations = self.reader.load_activations(sae_p1)
            tasks = [TrainingAutoEncoderTask(sae_p2, activations, consts.sae_p1_encoding_size, consts.sae_p2_encoding_size)]
            task_manager = TaskManager(tasks)
            task_manager.run_tasks()

            # Training third layer in SAE
            activations = self.reader.load_activations(sae_p2)
            tasks = [TrainingAutoEncoderTask(sae_p3, activations, consts.sae_p2_encoding_size, consts.sae_p3_encoding_size)]
            task_manager = TaskManager(tasks)
            task_manager.run_tasks()

    def _augment_data(self, images):
        cloned_images = dp.clone(images, 1)
        reflected_images = ip.apply_reflection(cloned_images)
        # Combine reflected and normal images
        images = dp.merge(images, reflected_images)
        images = dp.clone(images, 2)
        # Apply Random gaussian noise
        images = ip.apply_noise(images)
        return images

    def add_image(self, user_id, image):
        """
        params:
        user_id: string or int
        image: either a string of the filename or numpy array of an RGB image
        """
        im = Image(image)
        self.writer.save_image(user_id, im, consts.image_path)

    def remove_images(self):
        """
        Removes all images that have been persisted
        in consts.image_path
        """
        self.writer.remove_directory(consts.image_path)
        self.writer.create_directory(consts.image_path)
