import cv2
import skimage
import numpy as np
import argparse
import torch
import yaml
from pathlib import Path
import tqdm


import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import *


device = "cuda" if torch.cuda.is_available() else 'cpu'

colors = ((255,0,0), (0,255,0), (0,0,255), (255,255,0))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frame_scp", required=True)
    p.add_argument("--output", default='output.mp4')
    p.add_argument("--fps", default = 25)
    p.add_argument("--line", default = 400)
    return p.parse_args()


def get_frame(path):
    res = []
    with open(path, "r") as f:
        for line in f.readlines():
            res.append(line.replace("\n",""))
    return res


def main(args):

    os.makedirs(Path(args.output).parent, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(config['label_map'], 'r') as f:
        label_map = yaml.safe_load(f)
    
    
    # Load model
    if config['type'] == "MobileNet":
        model = MobileNet(len(label_map))
    else:
        model = ResNet50(len(label_map))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)

    frames = get_frame(args.frame_scp)

    line = args.line
    print(f"inferencing on num {len(frames)} frames")
    
    res, image = get_objects(frames, line)

    
    labels = predict(model, [i[-1] for i in res])

    video_writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (960, 540))
    for frame_idx in range(len(frames)):
        # Create a blank image
        img = skimage.io.imread(frames[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw the reference line at y=400
        cv2.line(img, (0, line), (960, line), (255, 255, 255), 2)

        # Draw objects if they are in the current frame range
        for i, obj in enumerate(res):
            start_f, end_f, y_start, y_end, _ = obj
            if start_f <= frame_idx <= end_f:
                cls, name = label_map[labels[i]]
                cv2.line(img, (  y_start,line), ( y_end ,line), colors[cls], 2)
                cv2.putText(img, name, (int((y_start + y_end)/2), line + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls], 2)
        # Write frame to video
        video_writer.write(img)
    video_writer.release()
    print(f"Video saved as {args.output}")



def get_objects(frames, line, thick = 2):
    """
    return a list of list of objects where each index stands for if the object is present or not.
    """
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    res = []
    image_fg = []
    image = []
    for f in frames:
        img = skimage.io.imread(f)
        img = img[line:line+thick]
        fg_mask = backSub.apply(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Noise removal with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        image_fg.append(fg_mask)
        image.append(img)
    image_fg = np.concatenate(image_fg, axis = 0)
    image = np.concatenate(image, axis = 0)
    contours, _ = cv2.findContours(image_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        y, x, w, h = cv2.boundingRect(cnt)
        if w > 40 and w < 400 and h < 200:
            ## Threshold
            time_start = int(x / thick)
            time_end = int((x+h)/thick)
            y_start = y
            y_end = y + w
            res.append([time_start, time_end, y_start, y_end, image[x:x+h, y:y+w]])
    return res, image

def predict(model, images, resize = True):
    model.eval()
    input = []
    if resize:
        for image in images:
            image = skimage.transform.resize(image, (224,224)) # [H,W,C]
            input.append(torch.from_numpy(image).permute(2,0,1).float().to(device)) # [C, H, W]
    input = torch.stack(input)
    with torch.no_grad():
        output = model(input)
    return torch.argmax(output, dim = 1).cpu().tolist()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
