import os
from PIL import Image
from pathlib import Path
# 将dsec数据集可视化的flow进行裁剪


flow_root = Path('/media/wqm/Data_files/DSEC/paper_vis/flow')
image_root = Path('/media/wqm/Data_files/DSEC/paper_vis/images')
saved_root = '/media/wqm/Data_files/DSEC/paper_vis/cropped'
sequence = ['zurich_city_11_b']
# 'thun_00_a', 'zurich_city_02_a',
region = (50, 2, 639, 340)
img_region = (50, 52, 639, 390)

for item in flow_root.iterdir():
    name = str(item.name)
    print(name)
    img = Image.open(item)
    cropped = img.crop(region)
    cropped.save(os.path.join(saved_root, name))

for item in image_root.iterdir():
    name = str(item.name)
    print(name)
    img = Image.open(item)
    cropped = img.crop(img_region)
    cropped.save(os.path.join(saved_root, name))