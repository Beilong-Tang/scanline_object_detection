import argparse
import os
import sys 
from pathlib import Path
import yaml
import tqdm
import random 
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from network import *
from dataset import ResizeDataset

def setup_seed(seed):
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type = str, required=True)
    return p.parse_args()


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    ## load yaml config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    ## setup seed
    setup_seed(config['seed'])

    ## Load class map
    with open(config['label_map'], "r") as f:
        label_map_dict = yaml.safe_load(f)

    ## load model
    if config['type'] == "MobileNet":
        if config.get('model_config') is not None:
            model = MobileNet(num_classes=len(label_map_dict), **config['model_config'])
        else:
            model = MobileNet(num_classes=len(label_map_dict))
    elif config['type'] == "ResNet50":
        if config.get('model_config') is not None:
            model = ResNet50(num_classes=len(label_map_dict), **config['model_config'])
        else:
            model = ResNet50(num_classes=len(label_map_dict))
    
    # ## Loading pretrained models
    # if args.pretrained is not None:
    #     print(f"Loading pretrained model from {args.pretrained}")
    #     state = torch.load(args.pretrained, map_location='cpu')
    #     model.load_state_dict(state)

    model.cuda()

    ## Optimzer
    if config['optim']['type'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config['optim']['lr'])

    ## Load data
    tr_dataset = ResizeDataset(config['tr_data'], config['label_map'], padding=config.get("padding"), augmentation=config.get("augmentation"))
    cv_dataset = ResizeDataset(config['cv_data'], config['label_map'], padding=config.get("padding"), augmentation=config.get("augmentation"))
    tr_data = DataLoader(tr_dataset, batch_size= config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    cv_data = DataLoader(cv_dataset, batch_size= config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    # get weighted loss
    if config.get("weighted_loss") is True:
        print("applying weighted cross entropy loss")
        loss_fn = torch.nn.CrossEntropyLoss(weight=tr_dataset.get_weights().cuda())
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    ## Save the ckpt
    ckpt_path = Path(args.config).absolute().parent / 'ckpt' / Path(args.config).stem
    os.makedirs(str(ckpt_path), exist_ok=True)
    print(f"saving model to {ckpt_path}")

    hist = {"train_epoch":[], "eval_epoch": [], "train_step":[]}
    ## Training
    for e in range(config['epochs']):
        model.train() # Allow training
        pbar = tqdm.tqdm(tr_data, desc=f"[Train] Epoch {e}", dynamic_ncols=True)
        
        # Train
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(pbar):
            ## calculate loss            
            img, label = data # [B, C, W, H], [B, 1]
            img, label = img.cuda(), label.cuda()
            label = label.squeeze(1) # [B]
            out = model(img)  # [B, N]
            loss = loss_fn(out, label)
            pred = torch.argmax(out, dim = 1) # [B]
            acc =(pred == label).sum().item()

            loss = loss / len(img)
            acc = acc / len(img)

            total_loss += loss.item()
            total_acc += acc
                
            # backward prop
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({'loss': f'{total_loss / (i+1):.3f}', 'accuracy':f'{total_acc / (i+1):.3f}'})
            hist['train_step'].append({'loss': f'{total_loss / (i+1):.3f}', 'accuracy':f'{total_acc / (i+1):.3f}'})
        hist['train_epoch'].append({"loss": (total_loss / len(tr_data)), "acc": (total_acc / len(tr_data))})
        # Eval
        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(cv_data, desc=f"[Eval] Epoch {e}", dynamic_ncols=True)
            loss = 0
            acc = 0
            acc_all = dict([(i, [0,0]) for i in range(0,len(label_map_dict))])
            for i, data in enumerate(pbar):
                img, label = data # [B, C, W, H], [B, 1]
                img, label = img.cuda(), label.cuda()
                label = label.squeeze(1) # [B]
                out = model(img)  # [B, N]
                loss += loss_fn(out, label) / len(img)
                pred = torch.argmax(out, dim = 1) # [B]
                acc += (pred == label).sum().item() / len(img)
                pbar.set_postfix({'loss': f'{(loss.item()/(i+1)):.3f}', 'accuracy':f'{(acc/(i+1)):.3f}'})

                ## Calculate total accuracy for each classification
                for i, _true_label in enumerate(label):
                    acc_all[_true_label.item()][1] +=1
                    if (pred[i] == _true_label):
                        acc_all[_true_label.item()][0] +=1

            for k, value in acc_all.items():
                acc_all[k] = value[0] / value[1]

            print(f"Eval epoch {e}: loss: {(loss.item()/(i+1)):.3f}, acc: {(acc/(i+1)):.3f}, {acc_all}")
        hist['eval_epoch'].append({"loss": loss.item() / (i+1), "acc": acc / (i+1), "acc_all": acc_all})
        torch.save(model.state_dict(), str(ckpt_path / f'epochs_{e}.pth'))
    torch.save(hist, str(ckpt_path / 'hist.pth'))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass