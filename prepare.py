import os
import torch
import cv2
import os
import numpy as np
from models.model import TRGAN
from params import *
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from torchvision import transforms

num_images = 15
height = 32
width = 192
directory = "./eval_files/rip4"
model_path = "files/iam_model.pth"
output_file_name = "/image-rip4-IAM.png"
text = "No two people can write precisely the same way just like no two people can have the same fingerprints"
output_path = "results"


def load_itw_samples(folder_path, num_samples=15):
    paths = glob.glob(f"{folder_path}/*.png")
    paths = np.random.choice(paths, num_samples, replace=len(paths) <= num_samples)

    imgs = []
    for i in paths:
        img = Image.open(i)
        imgs.append(np.array(img.convert("L")))

    imgs = [
        cv2.resize(imgs_i, (32 * imgs_i.shape[1] // imgs_i.shape[0], 32))
        for imgs_i in imgs
    ]

    imgs_pad = []
    imgs_wids = []

    trans_fn = get_transform(grayscale=True)

    for img in imgs:

        img = 255 - img
        img_height, img_width = img.shape[0], img.shape[1]
        outImg = np.zeros((img_height, width), dtype="float32")
        outImg[:, :img_width] = img[:, :width]

        img = 255 - outImg

        imgs_pad.append(trans_fn((Image.fromarray(img))))
        imgs_wids.append(img_width)

    imgs_pad = torch.cat(imgs_pad, 0)

    return imgs_pad.unsqueeze(0), torch.Tensor(imgs_wids).unsqueeze(0)


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


squeezed_torch = load_itw_samples(directory, num_images)

print("(2) Loading model...")

model = TRGAN()
model.netG.load_state_dict(
    torch.load(model_path, map_location="cpu", pickle_module=pickle)
)
print(model_path + " : Model loaded Successfully")

print("(3) Loading text content...")
text_encode = [j.encode() for j in text.split(" ")]
eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
eval_text_encode = eval_text_encode.to("cpu").repeat(batch_size, 1, 1)

os.makedirs(output_path, exist_ok=True)


def find_furthest_black_width_for_images(image_datas):
    print(image_datas[0][0])
    furthest_widths = []
    for data in image_datas[0][0]:
        furthest_width = 0
        for row in data:
            for i in range(191, 0, -1):
                if row[i] != 1:
                    if i > furthest_width:
                        furthest_width = i

        furthest_widths.append(furthest_width)
    return furthest_widths


swids = torch.tensor(find_furthest_black_width_for_images(squeezed_torch))

print(swids)
squeezed_swids = swids.unsqueeze(0)

page_val = model._generate_page(
    squeezed_torch[0].to(DEVICE),
    squeezed_swids,
    eval_text_encode,
    eval_len_text,
)

cv2.imwrite(output_path + output_file_name, page_val * 255)


print("\nOutput images saved in : " + output_path + output_file_name)
