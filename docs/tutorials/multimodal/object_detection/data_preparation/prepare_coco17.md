# AutoMM Detection - Prepare COCO2017 Dataset
:label:`sec_automm_detection_prepare_coco17`

[COCO](https://cocodataset.org/#home) is a large-scale object detection, segmentation, and captioning dataset. 
For detection, the most commonly used version is COCO2017 with the largest number of training data.
There are 80 classes, 123,287 images, 886,284 instances, and the median image ratio is 640 x 480.
This tutorial will walk through the steps of preparing this dataset for Autogluon MultiModalPredictor.

You need 42.7 GB disk space to download and extract this dataset. SSD is preferred over HDD because of its better performance.
The total time to prepare the dataset depends on your Internet speed and disk performance. For example, it often takes 20 min on AWS EC2 with EBS.

COCO has an [official download page](https://cocodataset.org/#download), 
but it's always easier to perform a one-step setup.
We prepared a bash script for one-step downloading the COCO17 dataset: 
[download_coco17.sh](https://raw.githubusercontent.com/awslabs/autogluon/master/examples/automm/object_detection/download_coco17.sh).
Or you can also use our cli tool `prepare_detection_dataset` that can download all datasets mentioned in our tutorials.
This python script is in our code: 
[prepare_detection_dataset.py](https://raw.githubusercontent.com/awslabs/autogluon/master/multimodal/src/autogluon/multimodal/cli/prepare_detection_dataset.py),
and you can also run it as a cli: `python3 -m autogluon.multimodal.cli.prepare_detection_dataset`.

### Download with Python Script

The python script does not show progress bar, but is promised to work on all major platforms.
If you are working on a Unix system and needs a progress bar, try the bash script!

You could either extract it in coco17 folder under current directory by running:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017
```

or extract it in coco17 folder under a provided output path:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name coco2017 --output_path ~/data
```

or make it shorter:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d coco17 -o ~/data
```

### Download with Bash Script

You could either extract it in coco17 folder under current directory by running:

```
bash download_coco17.sh
```

or extract it in coco17 folder under a provided output path:

```
bash download_coco17.sh ~/data
```

The command line output will show the progress bar:

```
extract data in ./coco17
--2022-11-02 20:19:49--  http://images.cocodataset.org/zips/train2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.18.68
Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.18.68|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 19336861798 (18G) [application/zip]
Saving to: ‘train2017.zip’

train2017.zip                          100%[=========================================================================>]  18.01G  27.0MB/s    in 7m 29s  

2022-11-02 20:27:18 (41.1 MB/s) - ‘train2017.zip’ saved [19336861798/19336861798]

--2022-11-02 20:27:18--  http://images.cocodataset.org/zips/val2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 54.231.171.137
Connecting to images.cocodataset.org (images.cocodataset.org)|54.231.171.137|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 815585330 (778M) [application/zip]
Saving to: ‘val2017.zip’

val2017.zip                            100%[=========================================================================>] 777.80M  43.0MB/s    in 20s     

2022-11-02 20:27:38 (39.2 MB/s) - ‘val2017.zip’ saved [815585330/815585330]

--2022-11-02 20:27:38--  http://images.cocodataset.org/zips/test2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 54.231.162.177
Connecting to images.cocodataset.org (images.cocodataset.org)|54.231.162.177|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 6646970404 (6.2G) [application/zip]
Saving to: ‘test2017.zip’

test2017.zip                           100%[=========================================================================>]   6.19G  42.3MB/s    in 2m 32s  

2022-11-02 20:30:11 (41.6 MB/s) - ‘test2017.zip’ saved [6646970404/6646970404]

--2022-11-02 20:30:11--  http://images.cocodataset.org/zips/unlabeled2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.71.116
Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.71.116|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 20126613414 (19G) [application/zip]
Saving to: ‘unlabeled2017.zip’

unlabeled2017.zip                       33%[========================>                                                 ]   6.37G  43.2MB/s    eta 5m 45s 
```

And after it finished, in coco17 folder it contains:

```
annotations  test2017  train2017  unlabeled2017  val2017
```

### The COCO Format

COCO also refers to the specific format (in `.json` file) the COCO dataset is using.
In Autogluon MultiModalPredictor, we strongly recommend using this as your data format.
Check :ref:`sec_automm_detection_convert_to_coco` and :ref:`sec_automm_detection_voc_to_coco` for more information
about create a COCO format dataset from scratch or from other format, especially VOC format.

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@misc{https://doi.org/10.48550/arxiv.1405.0312,
  doi = {10.48550/ARXIV.1405.0312},
  url = {https://arxiv.org/abs/1405.0312},
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Bourdev, Lubomir and Girshick, Ross and Hays, James and Perona, Pietro and Ramanan, Deva and Zitnick, C. Lawrence and Dollár, Piotr},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Microsoft COCO: Common Objects in Context},
  publisher = {arXiv},
  year = {2014},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
