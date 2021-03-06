import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import ExifTags
from tqdm import tqdm
import shutil

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']

files_create = ['data.names', 'data.shapes', 'data.txt', 'data_train.txt', 'data_test.txt', 'data_val.txt']

# to remove existing files 
def remove_existing_files(data_folder_path):

  for files in files_create:
    if os.path.exists(os.path.join(data_folder_path, files)):
      os.remove(os.path.join(data_folder_path, files))
  
  if os.path.exists(os.path.join(data_folder_path, "labels")):
    shutil.rmtree(os.path.join(data_folder_path, "labels"))


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


# def split_rows_simple(file='../data/sm4/out.txt'):  # from utils import *; split_rows_simple()
#     # splits one textfile into 3 smaller ones based upon train, test, val ratios
#     with open(file) as f:
#         lines = f.readlines()

#     s = Path(file).suffix
#     lines = sorted(list(filter(lambda x: len(x) > 0, lines)))
#     i, j, k = split_indices(lines, train_ratio, test=0.0 , validation_ratio)
#     for k, v in {'train': i, 'test': j, 'val': k}.items():  # key, value pairs
#         if v.any():
#             new_file = file.replace(s, '_' + k + s)
#             with open(new_file, 'w') as f:
#                 f.writelines([lines[i] for i in v])


def split_files(out_path, file_name, train_ratio, prefix_path=''):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train_ratio)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(out_path + '_' + key + '.txt', 'a') as file:
                for i in item:
                    file.write('%s%s\n' % (prefix_path, file_name[i]))


def split_indices(x, train_ratio, shuffle=True):  # split training data   
    test = 0.0
    validate = int(100 - train_ratio*100)/100
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train_ratio)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices



# def write_data_data(fname='data.data', nc=80):
#     # write darknet *.data file
#     lines = ['classes = %g\n' % nc,
#              'train =../out/data_train.txt\n',
#              'valid =../out/data_test.txt\n',
#              'names =../out/data.names\n',
#              'backup = backup/\n',
#              'eval = coco\n']

#     with open(fname, 'a') as f:
#         f.writelines(lines)


def image_folder2file(folder='images/'):  # from utils import *; image_folder2file()
    # write a txt file listing all imaged in folder
    s = glob.glob(folder + '*.*')
    with open(folder[:-1] + '.txt', 'w') as file:
        for l in s:
            file.write(l + '\n')  # write image list


# def add_coco_background(path='../data/sm4/', n=1000):  # from utils import *; add_coco_background()
#     # add coco background to sm4 in outb.txt
#     p = path + 'background'
#     if os.path.exists(p):
#         shutil.rmtree(p)  # delete output folder
#     os.makedirs(p)  # make new output folder

#     # copy images
#     for image in glob.glob('../coco/images/train2014/*.*')[:n]:
#         os.system('cp %s %s' % (image, p))

#     # add to outb.txt and make train, test.txt files
#     f = path + 'out.txt'
#     fb = path + 'outb.txt'
#     os.system('cp %s %s' % (f, fb))
#     with open(fb, 'a') as file:
#         file.writelines(i + '\n' for i in glob.glob(p + '/*.*'))
#     split_rows_simple(file=fb)


# def create_single_class_dataset(path='../data/sm3'):  # from utils import *; create_single_class_dataset('../data/sm3/')
#     # creates a single-class version of an existing dataset
#     os.system('mkdir %s_1cls' % path)


# def flatten_recursive_folders(path='../../Downloads/data/sm4/'):  # from utils import *; flatten_recursive_folders()
#     # flattens nested folders in path/images and path/JSON into single folders
#     idir, jdir = path + 'images/', path + 'json/'
#     nidir, njdir = Path(path + 'images_flat/'), Path(path + 'json_flat/')
#     n = 0

#     # Create output folders
#     for p in [nidir, njdir]:
#         if os.path.exists(p):
#             shutil.rmtree(p)  # delete output folder
#         os.makedirs(p)  # make new output folder

#     for parent, dirs, files in os.walk(idir):
#         for f in tqdm(files, desc=parent):
#             f = Path(f)
#             stem, suffix = f.stem, f.suffix
#             if suffix.lower() in img_formats:
#                 n += 1
#                 stem_new = '%g_' % n + stem
#                 image_new = nidir / (stem_new + suffix)  # converts all formats to *.jpg
#                 json_new = njdir / (stem_new + '.json')

#                 image = parent / f
#                 json = Path(parent.replace('images', 'json')) / str(f).replace(suffix, '.json')

#                 os.system("cp '%s' '%s'" % (json, json_new))
#                 os.system("cp '%s' '%s'" % (image, image_new))
#                 # cv2.imwrite(str(image_new), cv2.imread(str(image)))

#     print('Flattening complete: %g jsons and images' % n)


# def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
#     # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
#     # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
#     # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
#     # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
#     # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
#          None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#          51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
#          None, 73, 74, 75, 76, 77, 78, 79, None]
#     return x

def image_label_info(labels_folder, images_folder):
  labels_length = len(glob.glob1(labels_folder, "*.txt"))

  images_length = 0
  for formats in img_formats:
    formats = "*" + formats
    images_length += len(glob.glob1(images_folder, formats))
  print('Total labels = %d' % (labels_length))
  print('Total images = %d' % (images_length))

  if labels_length != images_length:
    print('Warning : Annotations of %d images are missing' % (images_length - labels_length))
