import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from copy import deepcopy  # for deep copy of the model

import matplotlib.pyplot as plt  # for plotting the training performance
import seaborn as sns  # for plotting the correlation matrix

# for encoding the categorical features and normalizing the numerical features
from sklearn.preprocessing import LabelEncoder, Normalizer

# for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

import torch  # for building the model
import torch.nn as nn  # for building the model
from torchtext.data.utils import get_tokenizer  # for tokenizing the text
from torchtext.vocab import build_vocab_from_iterator  # for building the vocabulary
from torch.utils.data import DataLoader  # for creating the data loader

# for evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load the dataset
df = pd.read_csv(filepath_or_buffer='amz_us_price_prediction_dataset.csv')
df.shape  # display the shape of the dataset

# combine the title and category columns
df["title"] = df["title"] + ". " + df["category"]
df = df[["title", "stars", "isBestSeller", "boughtInLastMonth",
         "price"]].copy()  # select the relevant columns
categories = "isBestSeller"  # define the categorical feature
le = LabelEncoder()  # instantiate the label encoder
# encode the categorical feature
df[categories] = le.fit_transform(df[categories])
# define the numerical features
numericals = ["stars", "boughtInLastMonth", "price"]
norm = Normalizer()  # instantiate the normalizer
# normalize the numerical features
df[numericals] = norm.fit_transform(df[numericals])
df.dropna(inplace=True)  # drop the missing values

tokenizer = get_tokenizer("basic_english")  # instantiate the tokenizer


def extract_tokens(x):  # define a function to extract the tokens
    for txt, _, _, _, _ in x:  # iterate through the dataset
        yield tokenizer(txt)  # yield the tokens


vocab = build_vocab_from_iterator(
    extract_tokens(df.values), specials=["<unk>"])  # build the vocabulary from the tokens
# set the default index to the unknown token
vocab.set_default_index(vocab["<unk>"])
def text_pipeline(x): return vocab(tokenizer(x))  # define the text pipeline


def collate(batch):  # define the collate function for the data loader
    text, numericals, target, offsets = [], [], [], [0]  # initialize the lists
    for txt, stars, seller, bought, price in batch:  # iterate through the batch
        processed_text = torch.tensor(text_pipeline(txt))  # process the text
        text.append(processed_text)  # append the processed text
        # append the numerical features
        numericals.append([stars, seller, bought])
        target.append(price)  # append the target
        offsets.append(processed_text.size(0))  # append the offset

    text = torch.cat(text)  # concatenate the text
    numericals = torch.tensor(numericals)  # convert the numericals to a tensor
    target = torch.tensor(target)  # convert the target to a tensor
    # convert the offsets to a tensor
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return text, numericals, target, offsets  # return the processed data


training_phase, testing = train_test_split(
    df.values, random_state=42, test_size=0.2)  # split the dataset into training and testing sets
# split the training set into training and validation sets
train, val = train_test_split(training_phase, random_state=42, test_size=0.3)

# define the batch size (batch is the number of samples that will be propagated through the network)
BATCH = 128
# define the learning rate (learning rate is a hyperparameter that controls how much we are adjusting the weights of our network with respect to the loss gradient)
LR = 0.1
# define the number of epochs (an epoch is a measure of the number of times all of the training vectors are used once to update the weights)
EPOCHS = 5

train_dl = DataLoader(train, batch_size=BATCH,
                      shuffle=True, collate_fn=collate)  # create the training data loader
val_dl = DataLoader(val, batch_size=BATCH, shuffle=True,
                    collate_fn=collate)  # create the validation data loader


class Price(nn.Module):  # define the model class
    def __init__(self, vocab_size, embed_size, feat_size):  # define the initialization method
        super(Price, self).__init__()  # initialize the model
        # initialize the embedding layer (embedding layer is a lookup table that maps the indices of the words to their corresponding word vectors)
        self.embed = nn.EmbeddingBag(vocab_size, embed_size)
        self.layer = nn.Sequential(nn.Linear(embed_size, 512),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(512),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(256),
                                   nn.Linear(256, feat_size),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(feat_size))  # initialize the linear layers (linear layer applies a linear transformation to the incoming data)

        self.fc = nn.Linear(feat_size*2, 1)

    # define the forward method (forward method defines the computation performed at every call)
    def forward(self, x, nums, off):
        x = self.embed(x, off)
        x = self.layer(x)
        x = torch.cat((x, nums), dim=1)
        x = self.fc(x)
        return nn.functional.relu(x)


vocab_size = len(vocab)  # get the size of the vocabulary
embed_size = 256  # set the size of the word vectors
feats = len(numericals)  # get the number of numerical features

# set the device to cuda if it is available, otherwise set it to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Price(vocab_size, embed_size, feats)  # instantiate the model
model = model.to(device)  # move the model to the device
# instantiate the optimizer (optimizer is a function that adjusts the weights of the network)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# instantiate the loss function (loss function computes the quantity that a model should seek to minimize during training)
criterion = nn.MSELoss()

best_model = deepcopy(model)  # make a deep copy of the model
best_loss = 10e9  # set the best loss to a very large number to ensure that the first validation loss will be the best
train_history = []  # initialize the training history
val_history = []  # initialize the validation history

for i in range(1, EPOCHS+1):  # iterate through the epochs
    model.train()  # set the model to training mode
    train_loss = 0.0
    train_total = 0
    for txt, nums, target, off in train_dl:  # iterate through the training data loader
        optimizer.zero_grad()
        if torch.cuda.is_available():
            txt, nums, target, off = txt.cuda(), nums.cuda(), target.cuda(), off.cuda()

        out = model(txt, nums, off)  # get the output from the model
        loss = criterion(out, target)  # compute the loss
        loss.backward()  # backpropagate the loss
        optimizer.step()  # update the weights
        train_loss += loss.item()  # add the loss to the training loss
        # add the number of samples to the training total
        train_total += out.size(0)
    train_loss = train_loss/train_total  # compute the average training loss
    # append the training loss to the training history
    train_history += [train_loss]

    model.eval()  # set the model to evaluation mode
    val_loss = 0.0
    val_total = 0
    with torch.no_grad():  # disable gradient computation to save memory
        for txt, nums, target, off in val_dl:  # iterate through the validation data loader
            if torch.cuda.is_available():
                txt, nums, target, off = txt.cuda(), nums.cuda(), target.cuda(), off.cuda()

            out = model(txt, nums, off)
            loss = criterion(out, target)
            val_loss += loss.item()
            val_total += out.size(0)

    val_loss = val_loss/val_total  # compute the average validation loss
    # append the validation loss to the validation history
    val_history += [val_loss]
    if val_loss < best_loss:  # check if the validation loss is the best loss
        best_model = deepcopy(model)
        best_loss = val_loss

    # print the training and validation loss
    print("Epoch {} train loss {} val loss {}".format(i, train_loss, val_loss))

epochs = list(range(1, EPOCHS+1))  # create a list of the epochs
plt.plot(epochs, train_history)  # plot the training history
plt.plot(epochs, val_history)  # plot the validation history
plt.legend(["training", "validation"])  # add the legend
plt.title("Training performance")  # add the title
plt.show()  # display the plot


def predict(x):  # define a function to make predictions
    txt, stars, seller, bought, price = x
    processed = torch.tensor(text_pipeline(txt))
    numericals = [[stars, seller, bought]]
    numericals = torch.tensor(numericals)
    off = torch.tensor([0])
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            processed, numericals, off = processed.cuda(), numericals.cuda(), off.cuda()
        out = model(processed, numericals, off)

    return out.item()  # return the prediction


predicted = []
real = []
for i in range(len(testing)):  # iterate through the testing set
    predicted += [predict(testing[i])]  # make predictions
    real += [testing[i, -1]]  # get the real values

mse = mean_squared_error(real, predicted)  # compute the mean squared error
mae = mean_absolute_error(real, predicted)  # compute the mean absolute error
r2 = r2_score(real, predicted)  # compute the r2 score

print("MSE: {}\nMAE: {}\nr2 score: {}".format(
    mse, mae, r2))  # print the evaluation metrics


### EXPLORE DATA ###
# df = pd.read_csv(filepath_or_buffer='amz_us_price_prediction_dataset.csv',
#                  index_col=['uid'])
# # df = df.copy()
# df = df[["reviews", "stars", "isBestSeller",
#          "boughtInLastMonth", "price"]].copy()
# df['has_reviews'] = df['reviews'] > 0
# df['taxe_free_income'] = df['price'] * df['boughtInLastMonth']

# # Pas concluant
# # df['stars_x_reviews'] = df['stars'] * df['reviews']

# corr_matrix = df.corr()
# print(corr_matrix["price"])
# print(corr_matrix["taxe_free_income"])

# # print(df.info())
# # print(df.head())
# # print(df.describe())

# # histogram(data_frame=df, x='stars', y='taxe_free_income',
# #           histfunc='avg', color='has_reviews')
# # histogram(data_frame=df, x='stars', y='reviews',
# #           histfunc='avg', color='has_reviews')
# # histogram(data_frame=df, x='stars', color='has_reviews')
# # histogram(data_frame=df, x='price', log_y=True)
# # histogram(data_frame=df, x='price',
# #           y='boughtInLastMonth', color='isBestSeller')
# # histogram(data_frame=df, x='taxe_free_income',
# #           y='price', histfunc='avg', log_y=True)
