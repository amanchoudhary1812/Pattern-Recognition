import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#Load an example image
image_path = 'Pattern-Recognition/Muki.jpeg'

image = Image.open("Pattern-Recognition/Muki.jpeg")

color_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    # grayscale with 20% probability
])

augmented_image = color_transforms(image)

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].imshow(image)
# ax[0].set-title("Original Image")

augmented_image.show()
