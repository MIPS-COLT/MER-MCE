import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds
import cv2


from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset



class FirstfaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Please describe the details of the facial expression in the picture.",
            "Can you provide a description of the facial expression shown by the person in the picture?",
            "What is the person's facial expression like in the picture?",
            "Could you give me some details about the person's facial expression captured in the picture?",
            "What words would you use to describe the facial micro expressions and actions of the characters in the photo?",
        ]
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)


    def __len__(self):
        return len(self.ann["annotations"])


    def __getitem__(self, index):
        info = self.ann["annotations"][index]

        image_file = '{}.jpg'.format(info['image_id'])

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption"]
        caption = self.text_processor(caption)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(random.choice(self.instruction_pool))
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }
    

class ECACEmotionDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        f = open(ann_path, 'r', encoding='utf-8').readlines()
        self.data = [json.loads(d) for d in f]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        input = self.data[index]['input']

        video_name = '{}.mp4'.format(self.data[index]['name'])
        video_path = os.path.join(self.vis_root, video_name)

        image = self.extract_frame(video_path)

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')

        image = self.vis_processor(image)

        caption = self.data[index]['target']        
        caption = self.text_processor(caption)
        instruction = "<Img><ImageHere></Img> [emotion] {} ".format(input)
        
        return {
            "name": self.data[index]['name'],
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }
    
    def extract_frame(self, video_path):
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        # Read the first frame
        success, frame = video_capture.read()

        if not success:
            raise ValueError("Failed to read video file:", video_path)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Release the video capture object
        video_capture.release()

        return frame_rgb
    


