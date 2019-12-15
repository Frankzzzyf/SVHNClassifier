# e4040-2019Fall-Project

Comprehensive review and TensorFlow implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf) 


### Dataset
<img src="https://github.com/potterhsu/SVHNClassifier/blob/master/images/inference1.png?raw=true" width="250">
<img src="https://github.com/potterhsu/SVHNClassifier/blob/master/images/inference2.png?raw=true" width="250">

> digit "10" means no digits

## Requirements

* Python 2.7
* Tensorflow
* h5py

    ```
    In Ubuntu:
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```

## Setup

1. Clone the source code

    ```
    $ git clone https://github.com/potterhsu/SVHNClassifier
    $ cd SVHNClassifier
    ```

2. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

3. Extract to data folder, now your folder structure should be like below:
    ```
    SVHNClassifier
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ```


## Usage

1. (Optional) Take a glance at original images with bounding boxes

    ```
    Open `draw_bbox.ipynb` in Jupyter
    ```

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ./data
    ```

1. (Optional) Test for reading TFRecords files

    ```
    Open `read_tfrecords_sample.ipynb` in Jupyter
    Open `donkey_sample.ipynb` in Jupyter
    ```

1. Train

    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train
    ```

1. Retrain if you need
    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train2 --restore_checkpoint ./logs/train/latest.ckpt
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ./data --checkpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

1. Visualize

    ```
    $ tensorboard --logdir ./logs
    ```

1. (Optional) Try to make an inference

    ```
    Open `inference_sample.ipynb` in Jupyter
    Open `inference_outside_sample.ipynb` in Jupyter
    $ python inference.py --image /path/to/image.jpg --restore_checkpoint ./logs/train/latest.ckpt
    ```

1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs/train2
    or
    $ rm -rf ./logs/eval
    ```
