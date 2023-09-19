import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open('resources.json', 'r') as file:
    data = json.load(file)
FILE = "data.pth"
loaded_state_dict = torch.load(FILE)
input_size = loaded_state_dict["input_size"]
hidden_size = loaded_state_dict["hidden_size"]
output_size = loaded_state_dict["output_size"]
all_words = loaded_state_dict["all_words"]
tags_list = loaded_state_dict["tags_list"]
model_state = loaded_state_dict["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = "Otavie"

def get_response(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags_list[predicted.item()]
    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]
    if probability.item() > 0.70:
        for intent in data["intents"]:
            if tag == intent["tag"]:
                # print(f'{bot_name}: {random.choice(intent["responses"])}')
                return random.choice(intent["responses"])
    else:
        # print(f"{bot_name}: I don not understand! I am just a chatbot 😂")
        return "I don not understand! I am just a chatbot 😂"
    

# if __name__ == '__main__':
#     print("Let's chat! Type 'quit' to exit!")
#     while True:
#         sentence = input("You: ")
#         if sentence == "quit":
#             break
#         get_response(sentence)
#         response = get_response(sentence)
#         print(response)