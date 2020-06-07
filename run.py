import argparse
import json
import cv2
import pandas as pd
from PIL import Image

from utils import *

# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(data_folder_path, json_file_dir, train_ratio):

    labels_folder = os.path.join(data_folder_path, "labels")
    images_folder = os.path.join(data_folder_path, "images")

    remove_existing_files(data_folder_path)  #remove file if it exist already

    os.makedirs(labels_folder)  # make new labels folder

    # Import json
    with open(json_file_dir) as f:
        data = json.load(f)

    # Write images and shapes
    name = os.path.join(data_folder_path, "data")
    file_id, file_name, width, height = [], [], [], []
    for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
        file_id.append(x['id'])
        file_name.append(os.path.join(data_folder_path, "images", x['file_name'].split('IMG_')[-1]))
        #file_name.append(os.path.join(data_folder_path, "labels", str(x['id']) + '.txt'))
        width.append(x['width'])
        height.append(x['height'])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % file_name[i])
 
        # shapes
        with open(name + '.shapes', 'a') as file:
            file.write('%g %g\n' % (x['width'], x['height']))

    # Write *.names file
    for x in tqdm(data['categories'], desc='Names'):
    
        with open(name + '.names', 'a') as file:
            file.write('%s\n' % x['name'])

    # Write labels file
    for x in tqdm(data['annotations'], desc='Annotations'):
        i = file_id.index(x['image_id'])  # image index
        label_name = Path(file_name[i]).stem + '.txt'
        
        # The Labelbox bounding box format is [top left x, top left y, width, height
        if x['bbox'] ==[]:
          x['bbox'] = [0,0,0,0]
          
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center

        if int(box[2]) != 0:
          box[[0, 2]] /= width[i]  # normalize x
          box[[1, 3]] /= height[i]  # normalize y
            
        #if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
            
            
        with open(os.path.join(labels_folder, label_name), 'a') as file:
            file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] , *box))
    
    image_label_info(labels_folder, images_folder)
    
    # Split data into train, test, and validate files
    split_files(name, file_name, train_ratio)               #split whole data for training and validation
    print('Done. Output saved to %s' % (data_folder_path))

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--json_file_dir', help="Directory path to json files.", type=str)
    parser.add_argument('--data_folder_path', help="path to data folder", type=str)
    #parser.add_argument('--image_folder_dir', help="Directory path to image folder.", type=str)
    parser.add_argument('--train_ratio', help="percentage of data for training. to split data into train and validation. It should be between 0 to 1 . Default it is 0.9", type=float, default= 0.9)

    opt = parser.parse_args()
    
    #convert_labelbox_json(name='ob', file='ob/coco_output.json')
    convert_labelbox_json(opt.data_folder_path, opt.json_file_dir, opt.train_ratio)
