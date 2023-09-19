import sys
import multiprocessing
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        def __len__(self):
            return self.n_samples

if __name__ == '__main__':
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn')
    with open("resources.json", "r") as file:
        data = json.load(file)
    all_words = []
    tags_list = []
    xy = []
    for intent in data['intents']:
        tag = intent['tag']
        tags_list.append(tag)
        for sentence in intent['patterns']:
            word = tokenize(sentence)
            all_words.extend(word)
            xy.append((word, tag))
    ignore_words = ['?', '!', ',', '.', ';', '/']
    all_words = ([stem(word) for word in all_words if word not in ignore_words])
    all_words = sorted(set(all_words))
    tags_list = sorted(tags_list)
    X_train = []
    Y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags_list.index(tag)
        Y_train.append(label)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    batch_size = 8
    hidden_size = 8
    output_size = len(tags_list)
    input_size = len(X_train[0])
    learning_rate = 0.001
    n_epoch = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(n_epoch):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words.to(device))
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'epoch {epoch+1}/{n_epoch}, loss={loss.item():.8f}')
    print(f'final loss, loss={loss.item():.8f}')
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags_list": tags_list
    }
    FILE = "data.pth"
    torch.save(data, FILE)

    print(f"Training complete. File saved to {FILE}")