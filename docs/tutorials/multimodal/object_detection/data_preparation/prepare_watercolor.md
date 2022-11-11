# AutoMM Detection - Prepare Watercolor Dataset
:label:`sec_automm_detection_prepare_watercolor`

[Watercolor](https://naoto0804.github.io/cross_domain_detection/) is a small object detection dataset with 1,000 training images and 1,000 testing images,
and has a specific domain, i.e. watercolor images. This dataset will be used to show how to \[Fast Finetune on Custom Dataset].

You need 7.5 GB disk space to download and extract this dataset. SSD is preferred over HDD because of its better performance.
The total time to prepare the dataset depends on your Internet speed and disk performance. For example, it often takes 8 min on AWS EC2 with EBS.

You can download the dataset from its [official project page](https://naoto0804.github.io/cross_domain_detection/).
We also prepared a bash script for one-step downloading the dataset: 
[download_watercolor.sh](https://github.com/awslabs/autogluon/blob/master/examples/automm/object_detection/download_watercolor.sh),
or use the python script that can download all datasets mentioned in our tutorials: [prepare_detection_dataset.py](https://github.com/awslabs/autogluon/blob/master/examples/automm/object_detection/prepare_detection_dataset.sh).

### Download with Python Script

The python script does not show progress bar, but is promised to work on all major platforms.
If you are working on a Unix system and needs a progress bar, try the bash script!

You could either extract it in coco17 folder under current directory by running:

```
python3 prepare_detection_dataset.py --dataset_name watercolor
```

or extract it in coco17 folder under a provided output path:

```
python3 prepare_detection_dataset.py --dataset_name watercolor --output_path ~/data
```

or make it shorter:

```
python3 prepare_detection_dataset.py -d watercolor -o ~/data
```

### Download with Bash Script
You could either extract it under current directory by running:

```
bash download_watercolor.sh
```

or extract it under a provided output path:

```
bash download_watercolor.sh ~/data
```

The command line output will show the progress bar:

```
% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                               Dload  Upload   Total   Spent    Left  Speed
4 3713M    4  170M    0     0  9627k      0  0:06:34  0:00:18  0:06:16 11.2M
```

And after it finished, VOC datasets are extracted in folder `watercolor`, it contains

```
Annotations  ImageSets  JPEGImages
```

### Dataset Format

Watercolor is in VOC format. **In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format instead.
Check :label:`sec_automm_detection_prepare_coco17` and :ref:`convert_data_to_coco_format` for more information
about COCO dataset and how to convert a VOC dataset to COCO.**

However, for fast proof testing we also have limit support for VOC format.
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages labels.txt
```

Here `labels.txt` shall be added manually to include all the labels in the dataset. 
For watercolor dataset, the content of `labels.txt` is shown as below:

```
bicycle
bird
car
cat
dog
person
```

In Unix system, you can also generate the `labels.txt` file automatically by running the following command in its root path (under folder `watercolor`)

```
grep -ERoh '<name>(.*)</name>' ./Annotations | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > labels.txt
```

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@inproceedings{inoue_2018_cvpr,
    author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
    title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```