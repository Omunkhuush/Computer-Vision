import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import os
import cv2
from get_loader import get_loader
from PIL import Image

modelPath = "./model.pt"
learning_rate = 0.00001
imagePath = "test_examples/"
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
train_loader, dataset = get_loader(
    root_folder='./dataset/images',
    annotation_file='./dataset/captions.txt',
    transform=transform,
    num_workers=4
)


def load_model(modelPath):
    model = torch.load(modelPath)
    # model = models.inception_v3(pretrained=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # checkpoint = torch.load(checkpointPath)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model


def load_images(imagePath):
    images = []
    if os.path.isfile(imagePath):
        path = imagePath.split("/")
        images.append(path[-1])
        x = ''
        for i in range(len(path)-1):
            x = x + str(path[i]) + '/'
        imagePath = x
    else:
        images = [f for f in os.listdir(
            imagePath) if os.path.isfile(os.path.join(imagePath, f))]
    return images, imagePath


def main():
    model = load_model(modelPath)
    images, imagesPath = load_images(imagePath)
    for i in images:
        img = transform(Image.open(imagesPath+i).convert("RGB")).unsqueeze(0)
        pred = model.caption_image(img.to(device), dataset.vocab)
        print("image name: {}".format(i), " ".join(pred))


main()
