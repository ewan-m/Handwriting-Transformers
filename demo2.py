import os
import time
from data.dataset import TextDataset, TextDatasetval
import torch
import cv2
import numpy as np
from models.model import TRGAN
from params import *
from torch import nn
from data.dataset import get_transform
import pickle
from PIL import Image
import tqdm
import shutil

text = "A paragraph is a series of related sentences developing a central idea, called the topic. Try to think about paragraphs in terms of thematic unity: a paragraph is a sentence or a group of sentences that supports one central, unified idea. Paragraphs add one idea at a time to your broader argument"
output_path = "results2"

model_path = "files/iam_model.pth"
data_path = "files/IAM-32.pickle"  # (iam)
# model_path = 'files/cvl_model.pth'; data_path = 'files/CVL-32.pickle' #(cvl)
# model_path = 'files/iam_model.pth'; data_path = 'files/CVL-32.pickle' #(iam-cvl-cross)
# model_path = 'files/cvl_model.pth'; data_path = 'files/IAM-32.pickle' #(cvl-iam-cross)#

print("(1) Loading dataset files...")

TextDatasetObjval = TextDatasetval(base_path=data_path, num_examples=15)
# print(TextDatasetObjval.IMG_DATA.keys())


# os.abort()

datasetval = torch.utils.data.DataLoader(
    TextDatasetObjval,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
    collate_fn=TextDatasetObjval.collate_fn,
)

print("(2) Loading model...")

model = TRGAN()

model.netG.load_state_dict(torch.load(model_path, map_location="cpu"))

print(model_path + " : Model loaded Successfully")

print("(3) Loading text content...")
text_encode = [j.encode() for j in text.split(" ")]
eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
eval_text_encode = eval_text_encode.to("cpu").repeat(batch_size, 1, 1)


if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

for i, data_val in enumerate(tqdm.tqdm(datasetval)):

    page_val = model._generate_page(
        data_val["simg"].to(DEVICE), data_val["swids"], eval_text_encode, eval_len_text
    )

    cv2.imwrite(output_path + "/image" + str(i) + ".png", page_val * 255)

print("\nOutput images saved in : " + output_path)
