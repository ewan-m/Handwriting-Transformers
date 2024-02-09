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


# Define the dimensions
num_images = 29
height = 32
width = 192

# Initialize an empty numpy array to hold the images
data = np.zeros((num_images, height, width), dtype=np.uint8)

# Load images from directory
directory = "./assets8"
image_files = sorted([file for file in os.listdir(directory) if file.endswith(".png")])

if len(image_files) != num_images:
    raise ValueError(
        "Number of images in the directory does not match the expected number."
    )

for i, image_file in enumerate(image_files):
    # Load image
    image_path = os.path.join(directory, image_file)
    image = Image.open(image_path)

    # Convert image to RGB and replace transparent parts with white
    image = image.convert("RGBA")
    image_data = np.array(image)
    alpha_channel = image_data[:, :, 3]  # Extract alpha channel
    image_data[:, :, :3][alpha_channel == 0] = [
        255,
        255,
        255,
    ]  # Replace transparent parts with white

    # Convert to grayscale
    image_data = np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Check dimensions
    if image_data.shape != (height, width):
        raise ValueError(f"Image {image_file} does not have the correct dimensions.")

    # Populate the numpy array
    data[i] = image_data


def plot_tensor(tensor):
    plt.imshow(tensor, cmap="gray", vmin=0, vmax=1)
    plt.show()


def find_furthest_black_width_for_images(image_datas):
    furthest_widths = []
    for data in image_datas:
        furthest_width = 0
        for row in data:
            for i in range(191, 0, -1):
                if row[i] != 254:
                    if i > furthest_width:
                        furthest_width = i

        furthest_widths.append(furthest_width)
    return furthest_widths


# Example usage:
furthest_widths = find_furthest_black_width_for_images(data)
print(furthest_widths)


model_path = "files/iam_model.pth"

text = "leg meeting production from Her ha writing take use find asset down clear proven assistants Legible handwriting when"
output_path = "results"

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


def plot_tensor(tensor):
    plt.imshow(tensor, cmap="gray", vmin=0, vmax=1)
    plt.show()


scaled_data = data.astype(np.float32) / 255.0

torch_data = torch.tensor(scaled_data)

squeezed_torch = torch_data.unsqueeze(0)

print(squeezed_torch)

swids = torch.tensor(find_furthest_black_width_for_images(data))
squeezed_swids = swids.unsqueeze(0)

page_val = model._generate_page(
    squeezed_torch.to(DEVICE),
    squeezed_swids,
    eval_text_encode,
    eval_len_text,
)

cv2.imwrite(output_path + "/image-8-IAM.png", page_val * 255)


print("\nOutput images saved in : " + output_path)
