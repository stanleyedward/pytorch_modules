import os

from timeit import default_timer as timer

import torch
from torch import nn
from torchvision import transforms
import data_preprocess, train_engine, build_model

import argparse #turn thesevraible into positional arguemnet flags!

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

#setup dir
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

#estup direvice agnostic ccode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#create trasfomrs
data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

#create data loaders and get class_names
train_dataloader, test_dataloader, class_names = data_preprocess.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transforms,
    batch_size=BATCH_SIZE
)

#create the model
model = build_model.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape = len(class_names)).to(device)
#setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
#start timer
start_time = timer()

#start training with help from engine.py
train_engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

end_time = timer()
print(f"[INFO] total training time: {end_time-start_time:.3f} seconds")
