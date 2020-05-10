import argparse
import json

import cv2
import pandas as pd
from PIL import Image

from utils import *

import shutil


# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(name, file, image_folder_dir):
    # Create folders
    path = make_folders()

    # Import json
    with open(file) as f:
        data = json.load(f)

    # Write images and shapes
    name = 'out' + os.sep + name
    file_id, file_name, width, height = [], [], [], []
    for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
        file_id.append(x['id'])
        file_name.append('\out\images\' + x['file_name'].split('IMG_')[-1])
        width.append(x['width'])
        height.append(x['height'])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % file_name[i])

        # shapes
        with open(name + '.shapes', 'a') as file:
            file.write('%g, %g\n' % (x['width'], x['height']))

    # Write *.names file
    for x in tqdm(data['categories'], desc='Names'):
        with open(name + '.names', 'a') as file:
            file.write('%s\n' % x['name'])

    # Write labels file
    for x in tqdm(data['annotations'], desc='Annotations'):
        i = file_id.index(x['image_id'])  # image index
        label_name = Path(file_name[i]).stem + '.txt'

        # The Labelbox bounding box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= width[i]  # normalize x
        box[[1, 3]] /= height[i]  # normalize y

        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
            with open('out/labels/' + label_name, 'a') as file:
                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] , *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))
    
    shutil.copyfile(image_folder_dir, os.getcwd() + '\out\images')

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--json_file_dir', help="Directory path to json files.", type=str)
    parser.add_argument('--folder_name', help="Name of the folder", type=str)
    parser.add_argument('--image_folder_dir', help="Directory path to image folder.", type=str)
    
    opt = parser.parse_args()
    
    #convert_labelbox_json(name='ob', file='ob/coco_output.json')
    convert_labelbox_json(opt.folder_name, opt.json_file_dir, opt.image_folder_dir)
