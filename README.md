# Scanline Traffic Object Detection (STOD)

Final project for DKU COMPSCI 302: Computer Vision

![demo](demo.gif)

## Goal:

Identify passing objects with only one line visible from traffic surveillance video.

## Requirements

```shell
git clone https://github.com/Beilong-Tang/scanline_object_detection.git
pip install -r requirements.txt
```

### Data Preparation

Download [UA-DETRAC](https://www.kaggle.com/datasets/thhyaa/uadetarc) and extract it to a specific folder.
(All the experiments are based on this dataset).

The UA-DETRAC dataset is a large-scale benchmark for vehicle detection and multi-object tracking in traffic surveillance videos. It was collected from real-world traffic scenarios in China under various weather and lighting conditions.

- 10 hrs (6 hrs training and 4 hrs validation)
- 25 fps from static camera
- 4 classes (car, bus, truck, others)

### Feature Extraction

To extract the data using central mapping algorithm. do 

```sh
## training set
python src/extract_data.py --data_path <tr_data_path> --meta_path conf/tr_data.csv \
    --output <output_dir>

## evaluation set
python src/extract_data.py --data_path <tr_data_path> --meta_path conf/val_data.csv \
    --output <output_dir>
```

This will give you `<output>/img` and `<output>/tr_data.scp` 
and `<output>/val_data.scp` where each line in the scp is formated as
`idx path label height width`. 

### Pretrained Models

For configs using pretrained models, you need to download them.

## File structure

`demo/` contains the visualization code I used to produce the results for the slides.
- Please change the `data_dir` of each notebook to be your own data path.

`src/` contains the source code for extracting passing objects, training and inference.


`exp/` contains the configs for each experiments.
- Note that you need to change the `tr_data` and `cv_data` based on your file path output from the Feature Extraction step. 

`ckpt/` contains the model checkpoints and history for each experiments.

`recipes/` contains the training scripts.

## Training


For instance, to train a mobilenet from scratch, you can do 
```sh
bash recipes/train_mobilenet_from_scratch.sh
```
(Make sure to change the data in the config, and also change the `CUDA_VISILBE_DEVICES` in the script)

## Inference

To run an inference using a trained model, you can do

```sh
python src/infer.py --config <config_yaml_path> \
    --ckpt <ckpt_path> \
    --frame_scp <frame_scp_path>
```

where `frame_scp` is a file containing all the frames. For instance
```
path/to/frame1.png
path/to/frame2.png
...
```
