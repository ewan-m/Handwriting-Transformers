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


# dict_keys(['0017', '0018', '0024', '0025', '0003', '0047', '0022', '0002', '0006', '0041', '0012', '0015', '0020', '0027', '0026', '0013', '0014', '0050', '0021', '0029', '0004', '0028', '0042', '0001', '0023', '0016', '0005'])

print(len(data["test"]["0001"]))


# Create directory to store PNG images
output_dir = "eval_files/rip4"
os.makedirs(output_dir, exist_ok=True)

# Save each image as PNG
for i, item in enumerate(data["test"]["0001"]):
    img = item["img"]
    label = item["label"]
    img.save(os.path.join(output_dir, f"image_{i}.png"))

# show(data["test"]["0017"][30]["img"])
