import discord
from discord.ext import commands
import keep_alive
import os


import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


def prefix_():
    return "!"

def client_(prefix):
    return commands.Bot(command_prefix=prefix)


prefix = "!"

client = client_(prefix)







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('C:\Python Codes 2\AI mit chat\intents.json', 'r') as f:
    intents = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

@client.event
async def on_message(msg):
    if msg.author != client.user:

        ctx = await client.get_context(msg)

        sentence = str(msg.content)
        print(sentence)
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print(prob.item())
        if prob.item() >= 0.70:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    await ctx.send(random.choice(intent["responses"]))
        









extens = ["Cogs.Events"]


for index,ext in enumerate(extens):
    client.load_extension(ext)
    print(index, ext)




keep_alive.keep_alive()

# client.run(os.getenv('TOKEN'))
client.run("ODIzOTUxNzI5Nzc1MjE0NTky.YFoS2A.rnWE_MmAOrb09aMvR_mL51VgFLU")