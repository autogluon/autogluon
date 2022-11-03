# Convert Data to MSCOCO Format

:label:`convert_data_to_coco_format`

MSCOCO is one of the most popular datasets for object detection
and its annotation format, usually referred to as the "MSCOCO format", has also been widely adopted.
The "MSCOCO format" is a json structure that governs how labels and metadata are formatted for a dataset.
We use MSCOCO format as the standard data format for training and inference in object detection tasks, and 
require that all data related to object detection tasks should conform to the "MSCOCO format".  
For details regarding MSCOCO dataset, please see this page.

## How to prepare MSCOCO format
### 1. Formatting folder Structure
Under the MSCOCO format, the overall folder structure of a dataset should follow:
```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        <imagename1>.<ext>
        <imagename2>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
        ...
```

### 2. Formatting ``*_labels.json``
Below are the key names and value definitions inside ``*_labels.json``:

```javascript
{
    "info": info,
    "licenses": [license], 
    "images": [image],  // list of all images in the dataset
    "annotations": [annotation],  // list of all annotations in the dataset
    "categories": [category]  // list of all categories
}

where:

info = {
    "year": int, 
    "version": str, 
    "description": str, 
    "contributor": str, 
    "url": str, 
    "date_created": datetime,
}

license = {
    "id": int, 
    "name": str, 
    "url": str,
}

image = {
    "id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int,  // the id of the license
    "date_captured": datetime,
}

category = {
    "id": int, 
    "name": str, 
    "supercategory": str,
}

annotation = {
    "id": int, 
    "image_id": int,  // the id of the image that the annotation belongs to
    "category_id": int,  // the id of the category that the annotation belongs to
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": int,  // 0 or 1,
}
```
For the sole purpose of running AutoMM, the fields ``"info"`` and ``"licenses"`` are optional. 
``"images"``, ``"categories"``, and ``"annotations"`` are required.



```json
{
  "info": {...},
"licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", 
            "id": 1, 
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ...
    ],
    "categories": [
        {"supercategory": "person", "id": 1, "name": "person"},
        {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
        ...
    ],
        
    "images": [
        {
            "license": 4, 
            "file_name": "<imagename0>.<ext>", 
            "height": 427, 
            "width": 640, 
            "date_captured": null, 
            "id": 397133
        },
        ...
    ],
    "annotations": [
        
        ...
    ]
}
```







The following is an example of one sample annotated with COCO format

## Converting Pascal VOC dataset to MSCOCO format
Pascal VOC is another very popular dataset for object detection, and it also follows its own data format. 
We provide an example script to easily convert VOC dataset to COCO format.

The script can be found in the following directory:
```
./examples/automm/object_detection/voc2coco.py
```

Assuming your VOC dataset has the following structure
```
<path_to_VOCdevkit>/
    VOC2007/
        ...
    VOC2012/
        ...
    ...
```

#### 1. Go to the script folder:
```
cd ./examples/automm/object_detection/
```

#### 2. Run the following command:
```
python voc2coco.py <path_to_VOCdevkit> \
-o <path_to_VOCdevkit>/<output_folder_name> \
--out-format coco
```

## Converting other formats to MSCOCO format
We have demonstrated the MSCOCO format and feel free to write your own code to convert your data into the MSCOCO format.
As long as your data conforms to MSCOCO format, it'll work perfectly with the AutoMM pipelines.
In addition, there are a number of 3rd party tools to convert data into MSCOCO format. 
For example, [FiftyOne](https://github.com/voxel51/fiftyone) provides functionalities to convert other formats such as CVAT, YOLO, 
and KITTI etc. into MSCOCO format.