import os
import cv2
import helpers.consts as consts
import matplotlib.pyplot as plt


def get_images(dir_path):
    files_names = os.listdir(dir_path)
    return [cv2.imread(f'{dir_path}/{file_name}', cv2.COLOR_BGR2GRAY) for file_name in files_names]


def show_images(images_data):
    inches_size = 10 * len(images_data)
    plt.gcf().set_size_inches(inches_size, inches_size)
    i = int(f'1{len(images_data)}0')
    for image_data in images_data:
        i += 1
        try:
            image = cv2.cvtColor(image_data[consts.IMAGE], cv2.COLOR_GRAY2RGB)
        except Exception:
            image = image_data[consts.IMAGE]
        plt.subplot(i)
        plt.imshow(image.astype('uint8'), cmap='gray')
        plt.title(image_data[consts.TITLE])
        plt.xticks([])
        plt.yticks([])


def create_algs_wrapper(map_alg_name_to_dict):
    def algs_wrapper(image, alg, **params):
        algs_dict = map_alg_name_to_dict[alg]
        current_alg = algs_dict[consts.ALG]
        default_params = algs_dict[consts.PARAMS]
        default_params.update(params)
        transformed_images = current_alg(image, **default_params)
        return transformed_images
    return algs_wrapper


def get_transformed_image_variants(image, algs_data, algs_wrapper):
    transformed_image_variants = []
    for alg_data in algs_data:
        args = alg_data.get(consts.PARAMS)
        transformed_image_variants.append({
            consts.IMAGE: algs_wrapper(
                image,
                alg_data[consts.ALG],
                **(args if args else {})
            ),
            consts.TITLE: alg_data[consts.ALG]
        })
    return transformed_image_variants
