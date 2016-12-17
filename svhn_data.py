import os
import struct
import numpy as np
import scipy
import sys
import tarfile
import PIL.Image as Image

from digit_struct import DigitStruct

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
from itertools import product
from six.moves.urllib.request import urlretrieve


DATA_PATH = "data/svhn/"
CROPPED_DATA_PATH = DATA_PATH+"cropped"
FULL_DATA_PATH = DATA_PATH+"full"
PIXEL_DEPTH = 255
NUM_LABELS = 10

OUT_HEIGHT = 64
OUT_WIDTH = 64
NUM_CHANNELS = 3
MAX_LABELS = 5

last_percent_reported = None


def read_data_file(file_name):
    file = open(file_name, 'rb')
    data = process_data_file(file)
    file.close()
    return data


def read_digit_struct(data_path):
    struct_file = os.path.join(data_path, "digitStruct.mat")
    dstruct = DigitStruct(struct_file)
    structs = dstruct.get_all_imgs_and_digit_structure()
    return structs


def extract_data_file(file_name):
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall(FULL_DATA_PATH)
    tar.close()


def convert_imgs_to_array(img_array):
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    chans = img_array.shape[2]
    num_imgs = img_array.shape[3]
    scalar = 1 / PIXEL_DEPTH
    # Note: not the most efficent way but can monitor what is happening
    new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    for x in range(0, num_imgs):
        # TODO reuse normalize_img here
        chans = img_array[:, :, :, x]
        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
        norm_vec = (255-chans)*1.0/255.0
        # Mean Subtraction
        norm_vec -= np.mean(norm_vec, axis=0)
        new_array[x] = norm_vec
    return new_array


def convert_labels_to_one_hot(labels):
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels


def process_data_file(file):
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0  # Fix for weird labeling in dataset
    labels_one_hot = convert_labels_to_one_hot(labels)
    img_array = convert_imgs_to_array(imgs)
    return img_array, labels_one_hot


def get_data_file_name(master_set, dataset):
    if master_set == "cropped":
        if dataset == "train":
            data_file_name = "train_32x32.mat"
        elif dataset == "test":
            data_file_name = "test_32x32.mat"
        elif dataset == "extra":
            data_file_name = "extra_32x32.mat"
        else:
            raise Exception('dataset must be either train, test or extra')
    elif master_set == "full":
        if dataset == "train":
            data_file_name = "train.tar.gz"
        elif dataset == "test":
            data_file_name = "test.tar.gz"
        elif dataset == "extra":
            data_file_name = "extra.tar.gz"
    else:
        raise Exception('Master data set must be full or cropped')
    return data_file_name


def make_data_dirs(master_set):
    if master_set == "cropped":
        if not os.path.exists(CROPPED_DATA_PATH):
            os.makedirs(CROPPED_DATA_PATH)
    elif master_set == "full":
        if not os.path.exists(FULL_DATA_PATH):
            os.makedirs(FULL_DATA_PATH)
    else:
        raise Exception('Master data set must be full or cropped')


def handle_tar_file(file_pointer):
    ''' Extract and return the data file '''
    print ("extract", file_pointer)
    extract_data_file(file_pointer)
    extract_dir = os.path.splitext(os.path.splitext(file_pointer)[0])[0]

    structs = read_digit_struct(extract_dir)
    data_count = len(structs)

    img_data = np.zeros((data_count, OUT_HEIGHT, OUT_WIDTH, NUM_CHANNELS),
                        dtype='float32')
    labels = np.zeros((data_count, MAX_LABELS+1), dtype='int8')

    for i in range(data_count):
        lbls = structs[i]['label']
        file_name = os.path.join(extract_dir, structs[i]['name'])
        top = structs[i]['top']
        left = structs[i]['left']
        height = structs[i]['height']
        width = structs[i]['width']
        if(len(lbls) < MAX_LABELS):
            labels[i] = create_label_array(lbls)
            img_data[i] = create_img_array(file_name, top, left, height, width,
                                           OUT_HEIGHT, OUT_WIDTH)
        else:
            print("Skipping {}, only images with less than {} numbers are allowed!").format(file_name, MAX_LABELS)

    return img_data, labels


def create_svhn(dataset, master_set):
    path = DATA_PATH+master_set
    data_file_name = get_data_file_name(master_set, dataset)
    data_file_pointer = os.path.join(path, data_file_name)

    if (not os.path.exists(data_file_pointer)):
        ''' Create the data dir structure '''
        print("Creating data directories")
        make_data_dirs(master_set)
    if os.path.isfile(data_file_pointer):
        if(data_file_pointer.endswith("tar.gz")):
            return handle_tar_file(data_file_pointer)
        else:
            ''' Use the existing file '''
            extract_data = read_data_file(data_file_pointer)
            return extract_data
    else:
        new_file = download_data_file(path, data_file_name)
        if(new_file.endswith("tar.gz")):
            return handle_tar_file(new_file)
        else:
            ''' Return the data file '''
            return read_data_file(new_file)


def download_progress(count, block_size, total_size):
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported = percent


def download_data_file(path, filename, force=False):
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    print "Attempting to download", filename
    saved_file, _ = urlretrieve(base_url + filename, os.path.join(path, filename), reporthook=download_progress)
    print "\nDownload Complete!"
    statinfo = os.stat(saved_file)
    if statinfo.st_size == get_expected_bytes(filename):
        print("Found and verified", saved_file)
    else:
        raise Exception("Failed to verify " + filename)
    return saved_file


def get_expected_bytes(filename):
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    elif filename == "test.tar.gz":
        byte_size = 276555967
    elif filename == "train.tar.gz":
        byte_size = 404141560
    elif filename == "extra.tar.gz":
        byte_size = 1955489752
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size


def train_validation_spit(train_dataset, train_labels):
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.1, random_state = 42)
    return train_dataset, validation_dataset, train_labels, validation_labels


def write_npy_file(data_array, lbl_array, data_set_name, data_path):
    np.save(os.path.join(DATA_PATH+data_path, data_path+"_"+data_set_name+'_imgs.npy'), data_array)
    print('Saving to %s_svhn_imgs.npy file done.' % data_set_name)
    np.save(os.path.join(DATA_PATH+data_path, data_path+"_"+data_set_name+'_labels.npy'), lbl_array)
    print('Saving to %s_svhn_labels.npy file done.' % data_set_name)


def load_svhn_data(data_type, data_set_name):
    # TODO add error handling here
    path = DATA_PATH + data_set_name
    imgs = np.load(os.path.join(path, data_set_name+'_'+data_type+'_imgs.npy'))
    labels = np.load(os.path.join(path, data_set_name+'_'+data_type+'_labels.npy'))
    return imgs, labels


def create_label_array(el):
    """[count, digit, digit, digit, digit, digit]"""
    num_digits = len(el)  # first element of array holds the count
    labels_array = np.ones([MAX_LABELS+1], dtype=int) * 10
    labels_array[0] = num_digits
    for n in range(num_digits):
        if el[n] == 10: el[n] = 0  # reassign 0 as 10 for one-hot encoding
        labels_array[n+1] = el[n]
    return labels_array


def create_img_array(file_name, top, left, height, width, out_height, out_width):
    img = Image.open(file_name)

    img_top = np.amin(top)
    img_left = np.amin(left)
    img_height = np.amax(top) + height[np.argmax(top)] - img_top
    img_width = np.amax(left) + width[np.argmax(left)] - img_left

    box_left = np.floor(img_left - 0.1 * img_width)
    box_top = np.floor(img_top - 0.1 * img_height)
    box_right = np.amin([np.ceil(box_left + 1.2 * img_width), img.size[0]])
    box_bottom = np.amin([np.ceil(img_top + 1.2 * img_height), img.size[1]])

    img = img.crop((box_left, box_top, box_right, box_bottom)).resize([out_height, out_width], Image.ANTIALIAS)
    pix = np.array(img)

    norm_pix = (255-pix)*1.0/255.0
    norm_pix -= np.mean(norm_pix, axis=0)
    return norm_pix


def generate_full_files():
    train_data, train_labels = create_svhn('train', 'full')
    train_data, valid_data, train_labels, valid_labels = train_validation_spit(train_data, train_labels)

    write_npy_file(train_data, train_labels, 'train', 'full')
    write_npy_file(valid_data, valid_labels, 'valid', 'full')

    test_data, test_labels = create_svhn('test', 'full')
    write_npy_file(test_data, test_labels, 'test', 'full')

    # extra_data, extra_labels = create_svhn('full', 'extra')
    print("Full Files Done!!!")


def generate_cropped_files():
    train_data, train_labels = create_svhn('train', 'cropped')
    train_data, valid_data, train_labels, valid_labels = train_validation_spit(train_data, train_labels)

    write_npy_file(train_data, train_labels, 'train', 'cropped')
    write_npy_file(valid_data, valid_labels, 'valid', 'cropped')

    test_data, test_labels = create_svhn('test', 'cropped')
    write_npy_file(test_data, test_labels, 'test', 'cropped')
    print("Cropped Files Done!!!")


if __name__ == '__main__':
    generate_cropped_files()
    generate_full_files()
