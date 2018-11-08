from PIL import Image
import numpy as np
import json


def get_pixel(img_dir):
    img = Image.open(img_dir)
    return np.asarray(img).tolist()


def digit_to_one_hot(corr_digit):
    arr = [0] * 10
    arr[corr_digit] = 1
    return arr


def get_normalized_image(img_pixel_arr):
    img_normal = []
    for pixel_line in img_pixel_arr:
        img_normal_line = [pixel / 255 for pixel in pixel_line]
        img_normal.append(img_normal_line)
    return img_normal


def get_1d_from_2d(arr2d):
    ret = []
    for arr1d in arr2d:
        for e in arr1d:
            ret.append(e)
    return ret


def save_to_json_file(filename, data):
    obj = open(filename, 'wb')
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
    obj.close()


if __name__ == "__main__":
    images = []
    correct_values = []

    root_dir = './data/mnist_png_testing/'
    for digit in range(0, 10):
        print(digit, "진행 중...")
        digit_dir = root_dir + str(digit) + '/'
        for file_number in range(1, 1001):
            file_dir = digit_dir + str(digit) + '_' + str(file_number) + '.png'
            img_arr = get_pixel(file_dir)

            normalized_pixel_img = get_normalized_image(img_arr)

            image_1d = get_1d_from_2d(normalized_pixel_img)
            correct_value = digit_to_one_hot(digit)

            images.append(image_1d)
            correct_values.append(correct_value)

    save_to_json_file(root_dir + 'images.json', images)
    save_to_json_file(root_dir + 'correctValues.json', correct_values)
