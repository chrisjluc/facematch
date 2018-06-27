import consts

from image import Image, NoFaceDetectedException

import fnmatch
import numpy as np
import shutil
import os
import uuid

class Storage(object):

    def __init__(self):
        self.create_directory(consts.data_path)
        self.create_directory(consts.image_path)
        self.create_directory(consts.model_path)

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def remove_directory(self, path):
        shutil.rmtree(path)


class Writer(Storage):

    def save_image(self, user_id, image, path, image_id=None):
        """
        Params:
        image: Image object
        user_id: also known as directory name

        Persists an Image object at the specified path in the
        directory named after the user_id.

        if image_id is passed, we use this else we generate
        a random uuid.
        """
        image.assert_valid_state()
        user_path = os.path.join(path, user_id)
        self.create_directory(user_path)
        if not image_id:
            image_id = str(uuid.uuid4())
        image_path = os.path.join(user_path, image_id)
        np.save(image_path + consts.image_ext, image.image)
        np.save(image_path + consts.landmarks_ext, image.landmark_points)
        np.save(image_path + consts.features_ext, image.feature_points)

    def save_model(self, model, name):
        """
        params:
        model: keras model
        name: file prefix

        Saves the model.
        """
        model_path = os.path.join(consts.model_path, name)
        json = model.to_json()
        open(model_path + consts.json_ext, 'w').write(json)
        model.save_weights(model_path + consts.h5_ext, overwrite=True)

    def save_activations(self, activations, model_name):
        model_path = os.path.join(consts.model_path, model_name)
        np.save(model_path + consts.activations_ext, activations)


class Reader(Storage):

    def get_images(self, user_id, path):
        """
        Given a user_id (also directory name) we load
        all the corresponding images into facematch.Image objects
        and return it as an array

        If there are .jpg images that aren't persisted as Image objects,
        we convert them into Image objects and save them.
        So next time we call get_images() we don't have to convert from jpg again.
        """
        user_path = os.path.join(path, user_id)
        if not os.path.exists(user_path):
            raise Exception('Path does not exist')

        image_ids = [f.replace(consts.image_ext, '')
                for f in os.listdir(user_path)
                if fnmatch.fnmatch(f, '*' + consts.image_ext)]

        # Convert jpg images into our image format if not already converted
        image_jpg_ids = [f.replace(consts.jpg_ext, '')
                for f in os.listdir(user_path)
                if fnmatch.fnmatch(f, '*' + consts.jpg_ext)]

        images_to_convert = set(image_jpg_ids) - set(image_ids)
        if images_to_convert:
            w = Writer()
            for image_id in images_to_convert:
                try:
                    im = Image(
                            os.path.join(user_path, image_id) + consts.jpg_ext)
                    w.save_image(user_id, im, path, image_id)
                    image_ids.append(image_id)
                except NoFaceDetectedException:
                    pass

        return [self._get_image(image_id, user_path) for image_id in image_ids]

    def _get_image(self, image_id, user_path):
        """
        Given an image_id (filename without extension)
        and the user path (directory) we load the image
        and return an Image object.
        """
        image_path = os.path.join(user_path, image_id)
        image = Image()
        image.image = np.load(image_path + consts.image_ext)
        image.landmark_points = np.load(image_path + consts.landmarks_ext)
        image.feature_points = np.load(image_path + consts.features_ext)
        image.assert_valid_state()
        return image

    def get_user_ids(self, path):
        """
        Returns all the user ids (directory names)
        within a certain directory at a specified path
        """
        return [os.path.basename(os.path.normpath(x[0]))
            for x in os.walk(path)
            if x[0] != path]

    def get_user_ids_by_descending_jpg_counts(self, path, count=None):
        """
        Returns all the user ids (directory names)
        within a certain directory at a specified path
        in descending order of jpg count.

        If count is None, return all the images
        """
        count_by_names_dict = {}
        for root, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                if dirname not in count_by_names_dict:
                    count_by_names_dict[dirname] = 0
                directory_path = os.path.join(path, dirname)
                for filename in os.listdir(directory_path):
                    if filename.endswith(consts.jpg_ext):
                        count_by_names_dict[dirname] += 1
        names_by_count = [x for x in count_by_names_dict.iteritems()]
        names_by_count.sort(key=lambda x: x[1], reverse=True)
        if count is not None:
            names_by_count = names_by_count[:count]
        return [name for name, count in names_by_count]

    def get_model(self, model_name):
        from keras.models import model_from_json

        model_path = os.path.join(consts.model_path, model_name)
        model_file = model_path + consts.json_ext
        weight_file = model_path + consts.h5_ext
        if not os.path.isfile(model_file):
            raise Exception('Model files do not exist')
        if not os.path.isfile(weight_file):
            raise Exception('Weight files do not exist')
        model = model_from_json(open(model_file).read())
        model.load_weights(weight_file)
        return model

    def load_activations(self, model_name):
        model_path = os.path.join(consts.model_path, model_name)
        return np.load(model_path + consts.activations_ext)
