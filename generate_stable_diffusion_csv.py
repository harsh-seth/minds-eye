import json
import pandas as pd
import os
import random
import shutil
import cv2
import numpy as np

img_paths = []
masks_paths = []
grid_img_paths = []
models_paths = []
text = []

data_dir = "pix3d_full"
data = json.load(open(os.path.join(data_dir, 'pix3d.json')))

for item in data:
    folder_dir = os.path.join(data_dir, os.path.dirname(item['model']))
    object_name = os.path.basename(folder_dir)
    img_path = os.path.join(data_dir, item['img'])
    mask_path = os.path.join(data_dir, item['mask'])
    model_path = os.path.join(data_dir, item['model'])
    grid_img_path = os.path.join('grid_images', f'{object_name}_grind.png')
    # text.append('3 by 3 grid of images from 9 different camera angles')
    item['text'] = '3 by 3 grid of images from 9 different camera angles'
    item['grid_images'] = os.path.join('grid_images', f'{object_name}_grid.png')
    # img_paths.append(os.path.join(os.path.basename(img_path)))
    # masks_paths.append(os.path.basename(mask_path))
    # grid_img_paths.append(os.path.basename(item['grid_images']))
    # models_paths.append(os.path.basename(model_path))


random.shuffle(img_paths)
random.shuffle(masks_paths)
random.shuffle(grid_img_paths)
random.shuffle(models_paths)
random.shuffle(data)

# data = data[:int((len(data)+1)*.50)]

train_data = data[:int((len(data)+1)*0.70)]
test_data = data[int((len(data)+1)*0.70):int((len(data)+1)*0.90)]
validation_data = data[int((len(data)+1)*0.90):]

for item in train_data:
    try:
        if os.path.exists(os.path.join(data_dir, item['grid_images'])):
            shutil.copy(os.path.join(data_dir, item['img']), os.path.join(data_dir, 'train'))
            img_paths.append(os.path.join('train', os.path.basename(item['img'])))
            # grid_image = cv2.imread(os.path.join(data_dir, item['grid_images']))
            # grid_image = np.array(grid_image).flatten().tolist()
            # grid_img_paths.append(grid_image)
            grid_img_paths.append(item['grid_images'])
            text.append('3 by 3 grid of images from 9 different camera angles')
    except:
        continue

for item in test_data:
    try:
        if os.path.exists(os.path.join(data_dir, item['grid_images'])):
            shutil.copy(os.path.join(data_dir, item['img']), os.path.join(data_dir, 'test'))
            img_paths.append(os.path.join('test', os.path.basename(item['img'])))
            # grid_image = cv2.imread(os.path.join(data_dir, item['grid_images']))
            # grid_image = np.array(grid_image).flatten().tolist()
            # grid_img_paths.append(grid_image)
            grid_img_paths.append(item['grid_images'])
            text.append('3 by 3 grid of images from 9 different camera angles')
    except:
        continue

for item in validation_data:
    try:
        if os.path.exists(os.path.join(data_dir, item['grid_images'])):
            shutil.copy(os.path.join(data_dir, item['img']), os.path.join(data_dir, 'valid'))
            img_paths.append(os.path.join('valid', os.path.basename(item['img'])))
            # grid_image = cv2.imread(os.path.join(data_dir, item['grid_images']))
            # grid_image = np.array(grid_image).flatten().tolist()
            # grid_img_paths.append(grid_image)
            grid_img_paths.append(item['grid_images'])
            text.append('3 by 3 grid of images from 9 different camera angles')
    except:
        continue


# with open(f'{data_dir}/metadata.jsonl', 'w') as fp:
#     for item in data:
#         fp.write(json.dumps(item) + '\n')

data = {
    'file_name' : img_paths,
    # 'masks' : masks_paths,
    'grid_images': grid_img_paths,
    # 'models': models_paths,
    'text': text
}
df = pd.DataFrame(data)

df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)