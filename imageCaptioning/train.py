import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_loader, dataset = get_loader(
        root_folder='./dataset/images',
        annotation_file='./dataset/captions.txt',
        transform=transform,
        num_workers=4
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    # hyperparameters
    embed_size = 512
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 50
    # tensorboard
    Writer = SummaryWriter('runs/flickr')
    step = 0
    # initialize model loss etc.
    model = CNNtoRNN(embed_size, hidden_size,
                     vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if load_model:
        step = load_checkpoint(torch.load(
            "my_checkpoint.pth.tar"), model, optimizer)
    model.train()
    for epoch in range(num_epochs):
        print("number of epoch: ", epoch)
        print_examples(model, device, dataset)
        if save_model:
            modelPath = 'model.pt'
            torch.save(model, modelPath)
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "step": step
            # }
            # save_checkpoint(checkpoint)
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            Writer.add_scalar("Training loss", loss.item(), global_step=epoch)
            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        print('Loss = ', loss)


if __name__ == "__main__":
    train()
