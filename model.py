import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import math
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_now_string
from tqdm import tqdm
from dataset import MyDataset

def generate_solution_csv(model_path, main_path="./"):
    with open(main_path + "processed/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    collection = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
            outputs = model(inputs)
            collection.append(outputs.detach().cpu().numpy())
    output_numpy = np.concatenate(collection, axis=0)
    print(output_numpy.shape)
    output_path = model_path.replace("saves/", "output/").replace(".pt", ".csv")
    df = pd.DataFrame(columns=["contest-tmp2m-14d__tmp2m", "index"])
    df["contest-tmp2m-14d__tmp2m"] = output_numpy.flatten()
    df["index"] = test_dataset.index
    df.to_csv(output_path, index=False)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(244, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        print("{} layers".format(len(self.fc)))

    def forward(self, x):
        return self.fc(x)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, total=len(dataloader)):
        inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


# def test(model, dataloader, device):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
#             output = model(inputs)
#             loss = criterion(output, labels)
#             running_loss += loss.item()
#     return running_loss / len(dataloader), math.sqrt(running_loss / len(dataloader))


def run(model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler, epochs, device, main_path):
    train_loss_record = []
    valid_loss_record = []
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        valid_loss = validate(model, valid_loader, criterion, device)
        # test_loss, test_loss_rmse = test(model, test_loader, device)
        timestring = get_now_string()
        train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        print("[{}] Epoch: {}, Train Loss: {:.9f}, Valid Loss: {:.9f}, lr: {:.9f}".format(timestring, epoch + 1, train_loss, valid_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step(valid_loss)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), main_path + "saves/model_{}.pt".format(timestring))
            print("model saved to {}".format(main_path + "saves/model_{}.pt".format(timestring)))
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(16, 9))
            plt.plot(range(1, len(train_loss_record) + 1), train_loss_record, label="train loss")
            plt.plot(range(1, len(valid_loss_record) + 1), valid_loss_record, label="valid loss")
            plt.legend()
            plt.show()
            plt.close()


if __name__ == "__main__":
    main_path = "./"

    with open(main_path + "processed/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(main_path + "processed/valid.pkl", "rb") as f:
        valid_dataset = pickle.load(f)
    with open(main_path + "processed/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))

    model = MyModel().to(device)
    # model.load_state_dict(torch.load(main_path + "saves/model_20230228_211049_069082.pt"))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    epochs = 100

    run(model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler, epochs, device, main_path)




