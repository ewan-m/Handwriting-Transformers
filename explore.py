import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os

with open("files/CVL-32.pickle", "rb") as f:
    data = pickle.load(f)


def show(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


print(len(data["test"]["0017"]))


# Create directory to store PNG images
output_dir = "eval_files/rip"
os.makedirs(output_dir, exist_ok=True)

# Save each image as PNG
for i, item in enumerate(data["test"]["0017"]):
    img = item["img"]
    label = item["label"]
    img.save(os.path.join(output_dir, f"image_{i}.png"))

# show(data["test"]["0017"][30]["img"])
