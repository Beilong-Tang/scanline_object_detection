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

## File structure

`demo/` contains the visualization code I used to produce the results for the slides.
- Please change the `data_dir` of each notebook to be your own data path.
