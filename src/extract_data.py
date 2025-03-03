import torch
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# import skimage
# import cv2
from ultralytics import YOLO
import glob
import cv2
import skimage
import json
from pathlib import Path
import re
import tqdm
import os
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import BoxObject, pad_list

IMG_HEIGHT = 540
IMG_WIDTH = 960

def read_anno(path):
    res = []
    if not os.path.exists(path):
        return res
    with open(path, "r") as f:
        for line in f.readlines():
            data = line.replace("\n","").split(" ")
            y = int(IMG_WIDTH * float(data[1]))
            x = int(IMG_HEIGHT * float(data[2]))
            width = int(IMG_WIDTH * float(data[3]))
            height = int(IMG_HEIGHT * float(data[4]))
            # Compute top-left corner
            x_start, y_start = x - height // 2, y - width // 2
            x_end, y_end = x_start + height, y_start + width
            res.append([y_start, x_start, y_end, x_end, -1, int(data[0])])
    return res

def get_tr_data(line: int , output_dir:str, frames:list[str], y_range = None, thick:int = 1, threshold=None, bw=False, use_yolo = True, anno_dir = None):
    """
    Applys Central Mapping Algorithm on dataset
    line: x-axis where the line is at
    if use_yolo is False, we will use the annotation files
    """
    os.makedirs(output_dir, exist_ok=True)
    if threshold is None:
        threshold = thick * 3
    save_idx = 0
    hist = [] # This list saves all the objects that are currently running
    res = []

    if use_yolo is False:
        print("using annotation")
        assert anno_dir is not None
    else:
        print("using yolo")
        model = YOLO(r"yolo11n.pt")
    
    if bw is True:
        print("using black and white")
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    for f in tqdm.tqdm(frames):
        img = skimage.io.imread(f)
        if bw:
            img = backSub.apply(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # img_line = img[line:line+1] # [1, W]
        ## Apply YoLo to detect the object
        if use_yolo is False:
            box = read_anno(str(Path(anno_dir) / (Path(f).stem + ".txt")))
        else:
            yolo_res = model(f, verbose = False)
            box = yolo_res[0].boxes.data
        box_list = [BoxObject(*i, img[line:line+thick]) for i in box]
        ## This list stores the objects that are crossing the line at the moment
        cur_obj = []
        for _box_obj in box_list:
            if _box_obj.x_start <= line and _box_obj.x_end >= line:
                ## It is on the line
                if y_range is None:
                    cur_obj.append(_box_obj)
                elif _box_obj.y_start > y_range[0] and _box_obj.y_end < y_range[1]:
                    cur_obj.append(_box_obj)                    
        if len(hist) == 0:
            ## If no History, we add history
            hist = cur_obj
            # print(f"here: {hist}")
        else:
            ## Append it to the history
            new_hist = []
            update = [False] * len(hist)
            for _obj in cur_obj:
                _center_x, _center_y = _obj.center # The center of the current object
                is_new = True # if the cur_obj is new
                for i, _hist_obj in enumerate(hist):
                    # print(_hist_obj)
                    if (
                        _center_x > _hist_obj.x_start
                        and _center_x < _hist_obj.x_end
                        and _center_y > _hist_obj.y_start
                        and _center_y < _hist_obj.y_end
                        ):
                        ## That's the new data
                        if _hist_obj.direction is None:
                            ## Check if this is going up or going down
                            if _obj.x_start > _hist_obj.x_start:
                                # Goes down
                                _hist_obj.direction = 0
                            else:
                                # Goes up
                                _hist_obj.direction = 1
                        ## Add that to data
                        if _hist_obj.direction == 1:
                            _obj.objs =_hist_obj.objs + _obj.objs
                        else:
                            _obj.objs =_obj.objs + _hist_obj.objs
                        _obj.direction = _hist_obj.direction
                        hist[i] = _obj
                        update[i] = True
                        is_new = False
                        break
                if is_new: 
                    ## This means this object is new
                    new_hist.append(_obj)
            ## Remove the history as compeleted and save them to dir
            res_hist = []
            for i, _h in enumerate(hist):
                if update[i]:
                    res_hist.append(_h)
                else:
                    ## Save them
                    _h.objs = pad_list(_h.objs, 0, mode=args.pad_mode)
                    merged = np.concatenate(_h.objs).astype(np.uint8) # [T, D, C]
                    if merged.shape[0] >= threshold:
                        save_path = f"{output_dir}/{save_idx}.png"
                        skimage.io.imsave(save_path, merged)
                        res.append([save_idx, f"{Path(save_path).absolute()}", _h.cls, merged.shape[0], merged.shape[1]])
                        save_idx+=1
                    pass 
            hist = res_hist + new_hist
    return res



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help = "the base dir for UA-DETRAC dataset")
    p.add_argument("--meta_path", type= str, required=True, help="The meta .csv data of the data to process")
    p.add_argument("--thick", type= int, default = 2, help = "the thickness of the line")
    p.add_argument("--output", type= str, required=True, help="The output_path")
    p.add_argument("--bw", action='store_true')
    p.add_argument("--use_yolo",action='store_true')
    p.add_argument("--anno_dir", type= str, default = None)
    p.add_argument("--pad_mode", type= str, default = 'constant')
    return p.parse_args()


def main(args):
    ### Extract Train Dataset
    train_scp = {}
    pattern = r'MVI_(\d+)_img'
    # for i in tqdm.tqdm(glob.glob(f"/DKUdata/tangbl/courses/CS302_CV/final_project/data/content/UA-DETRAC/DETRAC_Upload/images/train/*.jpg")):
    for i in tqdm.tqdm(glob.glob(f"{args.data_path} / *.jpg")):
        match = re.search(pattern, Path(i).stem)
        video_id = match.group(1)
        if train_scp.get(video_id) is None:
            train_scp[video_id] = [i]
        else:
            train_scp[video_id].append(i)

    out_base = Path(args.output).absolute()
    os.makedirs(str(out_base / 'img'), exist_ok = True)
    out_scp = out_base / (Path(args.meta_path).stem + ".scp")
    scp = open(out_scp, "w")
    # Load data
    csv = pd.read_csv(args.meta_path)
    cls_res = {}
    with open(train_scp, "r") as file:
        train_scp = json.load(file)
    for idx, row in csv.iterrows():
        print(f"Running data {row['id']} : {idx+1}/{len(csv)} ")
        res = get_tr_data(int(row['x']), 
                          str(out_base / 'img' / str(row['id'])), 
                          train_scp.get(str(row['id'])), 
                          (int(row['y_start']), int(row['y_end'])),
                          thick=args.thick,
                          bw = args.bw,
                          use_yolo = args.use_yolo,
                          anno_dir= args.anno_dir)
        # Iterate the res
        for idx, path, cls, shape1, shape2 in res:
            if cls_res.get(cls) is None:
                cls_res[cls] = 1
            else:
                cls_res[cls] +=1
            scp.write(f"{row['id']}_{row['type']}_{idx} {cls} {path} {shape1} {shape2}\n")
            pass
    json.dump(cls_res, open(str(out_base /(Path(args.meta_path).stem + "_stats.json") ), "w"), indent=2)
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
