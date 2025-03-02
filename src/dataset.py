import yaml
import torch
import random
from torch.utils.data import Dataset
import skimage
import numpy as np 
    


def pad_center(img, target_size):
    """
    Pads an image to make it square and then resizes it to the target size.

    Args:
        img (numpy array): Input image of shape (H, W, C).
        target_size (tuple): Desired output size (H, W).

    Returns:
        numpy array: Resized image with padding.
    """
    ## if img shape > target_size
    h, w, c = img.shape
    if h > target_size[0] or w > target_size[1]:
        return skimage.transform.resize(img, target_size, mode="reflect", anti_aliasing=True)

    # Compute padding (to center the image)
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2

    # Apply padding (mode='constant' adds black, 'reflect' mirrors edges)
    img_padded = np.pad(img, ((pad_h, target_size[0] - h - pad_h), (pad_w, target_size[1] - w - pad_w), (0, 0)), mode='constant')

    # Resize to target size
    # img_resized = skimage.transform.resize(img_padded, target_size, mode='reflect', anti_aliasing=True)

    return img_padded

class ResizeDataset(Dataset):

    def __init__(self, scp_file, map_file, restrict=None, resize=(224,224), padding = None, augmentation = None):
        """
        restrict can be {2:40} That means we restrict label 2 to be 40
        if padding is False, use super resolution, else use padding.

        """
        
        with open(map_file, "r") as f:
            map_dict = yaml.safe_load(f)

        scp_dict = {} # {cls: List[path]}
        with open(scp_file, "r") as f:
            for line in f.readlines():
                content = line.replace("\n","").split(" ")
                path = content[2]
                cls = map_dict[int(content[1])]
                if scp_dict.get(cls) is None: 
                    scp_dict[cls] = [path]
                else:
                    scp_dict[cls].append(path)
        scp_dict = dict(sorted(scp_dict.items()))

        if restrict is not None:
            ## Randomly sampling data. Note that in this version, restrict should always be true
            for k, v in restrict.items():
                random.shuffle(scp_dict[k])
                scp_dict[k] = scp_dict[k][:v]
        
        scp = []
        for key, v in scp_dict.items():
            for i in v:
                scp.append([i, int(key), False])
                if augmentation is not None:
                    ## Apply augmentations here
                    if key in augmentation:
                        scp.append(i, int(key), True)

        log_info={}
        for _, cls, _ in scp:
            if log_info.get(cls) is None:
                log_info[cls] = 1
            else:
                log_info[cls] +=1
        self.log_info = dict(sorted(log_info.items()))
        print(f"Training data info:  {log_info}")
        
        self.scp_dict = scp_dict
        self.scp = scp
        self.resize = resize
        self.padding = padding
        print(f"apply padding: {self.padding}")
    
    def get_weights(self):
        class_counts = torch.tensor([i for i in self.log_info.values()])
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights / class_weights.sum()  # Normalize
        return class_weights

    def __len__(self):
        return len(self.scp)

    def __getitem__(self, index):
        path, cls, flip = self.scp[index]
        img = skimage.io.imread(path) # [H, W, C]
        if flip:
            ## Apply flipping
            img = np.fliplr(img)
        if self.padding is not None:
            img = pad_center(img, (224,224))
        else:
            img = skimage.transform.resize(img, self.resize, mode='reflect', anti_aliasing=True) # [H,W,C]
        img = skimage.util.img_as_float(img)
        return torch.from_numpy(img).permute(2,0,1).float(), torch.tensor([cls], dtype = torch.long)


def collate_fn(data):
    return data
