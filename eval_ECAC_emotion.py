import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.registry import registry

from minigpt4.datasets.datasets.first_face import ECACEmotionDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='ECAC_emotion_caption', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

model, vis_processor = init_model(args)

model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

save_path = cfg.run_cfg.save_path

text_processor_cfg = cfg.datasets_cfg.ECAC_emotion_caption.text_processor.train
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
vis_processor_cfg = cfg.datasets_cfg.ECAC_emotion_caption.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


print(args.dataset)
if 'ECAC_emotion_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["ECAC_emotion_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["ECAC_emotion_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["ECAC_emotion_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ECAC_emotion_caption"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = ECACEmotionDataset(vis_processor, text_processor, img_path, eval_file_path)
    # print(data)
    # print(data[0])
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    targets_list = []  
    answers_list = []  
    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        targets = batch['answer']

        texts = prepare_texts(instruction_input, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for i in range(len(answers)):
            if answers[i] not in ['neutral', 'sadness', 'fear', 'disgust', 'surprise', 'anger', 'joy']:
                answers[i] = 'neutral'
        # print("targets: ", targets)
        # print("answers: ", answers)
        targets_list.extend(targets)  
        answers_list.extend(answers)  



    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted')
    recall = recall_score(targets_list, answers_list, average='weighted')
    f1 = f1_score(targets_list, answers_list, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    confusion_mat = confusion_matrix(targets_list, answers_list)
    print(confusion_mat)

# torchrun  --nproc_per_node 1 eval_ECAC_emotion.py --cfg-path eval_configs/minigptv2_eval_ECAC_emotion.yaml