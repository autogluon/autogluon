"""
Pascal VOC format utilities for object detection.
Provides functions for handling VOC XML files and annotations.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Final, List, Optional, Tuple

import defusedxml.ElementTree as ET
import numpy as np

from .format_converter import dump_voc_classes

logger = logging.getLogger(__name__)


# VOC Dataset Structure
VOC_DIRECTORIES: Final[List[str]] = [
    "Annotations",
    "ImageSets",
    "ImageSets/Main",
    "JPEGImages"
]


def dump_voc_xml_files(
    voc_annotation_path: str, 
    voc_annotation_xml_output_path: Optional[str] = None
) -> List[str]:
    """
    Extract and optionally save list of XML annotation files from VOC dataset.

    Args:
        voc_annotation_path: Root path for VOC format annotations
        voc_annotation_xml_output_path: Optional path to save list of XML files

    Returns:
        List of XML file paths
    """
    if not os.path.exists(voc_annotation_path):
        raise ValueError("Path does not exist: {}".format(voc_annotation_path))

    files = os.listdir(voc_annotation_path)
    annotation_path_base_name = os.path.basename(voc_annotation_path)
    
    # Collect XML files
    xml_file_paths = [
        os.path.join(annotation_path_base_name, f)
        for f in files
        if f.endswith(".xml")
    ]

    if not xml_file_paths:
        warnings.warn(f"No XML files found in {voc_annotation_path}")

    # Save file list if requested
    if voc_annotation_xml_output_path:
        with open(voc_annotation_xml_output_path, "w") as f:
            f.write("\n".join(xml_file_paths))
        logger.info("Saving results to: {}".format(voc_annotation_xml_output_path))

    return xml_file_paths


def process_voc_annotations(
    voc_annotation_path: str,
    voc_class_names_output_path: str,
    voc_annotation_xml_output_path: str
) -> None:
    """
    Process VOC annotations to extract class names and XML file paths.

    Args:
        voc_annotation_path: Root path for VOC format annotations
        voc_class_names_output_path: Path to save unique class names
        voc_annotation_xml_output_path: Path to save XML file paths

    Raises:
        ValueError: If annotation path doesn't exist
    """
    if not os.path.exists(voc_annotation_path):
        raise ValueError("Path does not exist: {}".format(voc_annotation_path))

    # Get unique class names
    class_names = dump_voc_classes(
        voc_annotation_path=voc_annotation_path,
        voc_class_names_output_path=voc_class_names_output_path
    )
    
    # Get XML file paths
    xml_files = dump_voc_xml_files(
        voc_annotation_path=voc_annotation_path,
        voc_annotation_xml_output_path=voc_annotation_xml_output_path
    )
    
    logger.info(
        "Processed VOC annotations: %d classes, %d XML files",
        len(class_names),
        len(xml_files)
    )


def save_result_voc_format(predictions: List[dict], result_path: str) -> None:
    """
    Save detection results in VOC format.

    Args:
        predictions: List of prediction dictionaries
        result_path: Path to save results
    """
    # Ensure result path has .npy extension
    result_name, _ = os.path.splitext(result_path)
    result_path = result_name + ".npy"
    
    # Save predictions
    np.save(result_path, predictions)
    logger.info("Saving results to: {}".format(result_path))


def verify_voc_dataset(root_path: str) -> Tuple[bool, List[str]]:
    """
    Verify the structure and contents of a VOC dataset.

    Args:
        root_path: Root directory of VOC dataset

    Returns:
        Tuple containing:
            - Boolean indicating if dataset is valid
            - List of missing or invalid components
    """
    root_path = Path(root_path)
    missing_components = []

    # Check required directories
    for directory in VOC_DIRECTORIES:
        dir_path = root_path / directory
        if not dir_path.is_dir():
            missing_components.append(f"Missing directory: {directory}")

    # Check Annotations directory has XML files
    anno_dir = root_path / "Annotations"
    if anno_dir.is_dir():
        xml_files = list(anno_dir.glob("*.xml"))
        if not xml_files:
            missing_components.append("No XML files found in Annotations directory")
    
    # Check JPEGImages directory has images
    images_dir = root_path / "JPEGImages"
    if images_dir.is_dir():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        if not image_files:
            missing_components.append("No JPEG images found in JPEGImages directory")

    # Verify matching annotations and images
    if anno_dir.is_dir() and images_dir.is_dir():
        xml_stems = {f.stem for f in anno_dir.glob("*.xml")}
        img_stems = {f.stem for f in images_dir.glob("*.jp*g")}
        
        unmatched_xml = xml_stems - img_stems
        unmatched_img = img_stems - xml_stems
        
        if unmatched_xml:
            missing_components.append(
                f"Found {len(unmatched_xml)} XML files without matching images"
            )
        if unmatched_img:
            missing_components.append(
                f"Found {len(unmatched_img)} images without matching XML files"
            )

    return len(missing_components) == 0, missing_components


def validate_voc_xml(xml_path: str) -> Tuple[bool, List[str]]:
    """
    Validate contents of a VOC XML annotation file.

    Args:
        xml_path: Path to XML file

    Returns:
        Tuple containing:
            - Boolean indicating if XML is valid
            - List of validation errors
    """
    errors = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check required elements
        required_elements = ['filename', 'size', 'width', 'height']
        for elem in required_elements:
            if root.find(elem) is None:
                errors.append(f"Missing required element: {elem}")

        # Validate size elements
        size = root.find('size')
        if size is not None:
            try:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                if width <= 0 or height <= 0:
                    errors.append("Invalid image dimensions")
            except (ValueError, AttributeError):
                errors.append("Invalid size format")

        # Validate object annotations
        for obj in root.iter('object'):
            name = obj.find('name')
            if name is None or not name.text:
                errors.append("Object missing class name")
                
            bbox = obj.find('bndbox')
            if bbox is not None:
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    if xmin >= xmax or ymin >= ymax:
                        errors.append(f"Invalid bounding box: {xmin},{ymin},{xmax},{ymax}")
                except (ValueError, AttributeError):
                    errors.append("Invalid bounding box format")
            else:
                errors.append("Object missing bounding box")

    except ET.ParseError as e:
        errors.append(f"XML parsing error: {str(e)}")
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    return len(errors) == 0, errors
