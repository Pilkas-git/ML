import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import glob
import torch
from config.train_config import cfg
from PIL import Image

class OpenImagesDataset:

    def __init__(self, root,
                 transform=None,
                 dataset_type="train"):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.min_image_num = -1
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        width, height = image.size
        boxes = copy.copy(image_info['boxes'])
        boxes[:, 0] *= height
        boxes[:, 1] *= width
        boxes[:, 2] *= height
        boxes[:, 3] *= width
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)

        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __getitem__(self, index):
        image, target = self._getitem(index)
        return image, target

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = f"{self.root}/{self.dataset_type}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        annotations = annotations.loc[annotations['LabelName'].isin(cfg.class_Labels)]

        class_names = ['BACKGROUND'] + sorted(list(annotations['LabelName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            fileLocation = glob.glob(f"{self.root}/*/{image_id}.jpg")
            if not fileLocation:
                continue
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            labels = np.array([class_dict[name] for name in group["LabelName"]], dtype='int64')
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = glob.glob(f"{self.root}/*/{image_id}.jpg")
        image = Image.open(str(image_file[0]))
        return image

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))