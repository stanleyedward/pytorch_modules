"""
training and testing model
"""
import torch
import time
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.optim.lr_scheduler
from torch.utils.tensorboard import SummaryWriter

def train_step(epoch: int,
               model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler,
               device: torch.device.Device,
               disable_progress_bar: bool = False) -> Tuple[float, float]:
  """trains a pytorch model for a single epoch.
  (forward pass, loss calculation, optimizer step).

  Args:
    model: torch.nn.Module.
    dataloader: torch.utils.data.DataLoader
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device

  Returns:
    tuple(train_loss, train_accuracy). 
    for example:
    (0.1112, 0.8743)
  """
  model.train()
  train_loss, train_acc = 0, 0
  
  # loop through data loader data batches
  progress_bar = tqdm(
        enumerate(dataloader), 
        desc=f"Training Epoch {epoch}", 
        total=len(dataloader),
        disable=disable_progress_bar
    )

  for batch, (X, y) in progress_bar:
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if lr_scheduler:
        lr_scheduler.step()
      

      # calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

      # update progress bar
      progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1),
            }
        )

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(epoch: int,
              model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              disable_progress_bar: bool = False) -> Tuple[float, float]:
  """tests a pytorch model for a single epoch.
  performs a forward pass on a testing dataset.

  Args:
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    loss_fn: torch.nn.Module
    device: torch.device

  Returns:
    tuple(test_loss, test_accuracy). 
    for example:
    (0.0223, 0.8985)
  """
  model.eval() 
  test_loss, test_acc = 0, 0

  # loop through data loader data batches
  progress_bar = tqdm(
      enumerate(dataloader), 
      desc=f"Testing Epoch {epoch}", 
      total=len(dataloader),
      disable=disable_progress_bar
  )

  # inference context manager
  with torch.inference_mode(): 
      # loop through dataLoader batches
      for batch, (X, y) in progress_bar:
          X, y = X.to(device), y.to(device)
          test_pred_logits = model(X)
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

          # update progress bar
          progress_bar.set_postfix(
              {
                  "test_loss": test_loss / (batch + 1),
                  "test_acc": test_acc / (batch + 1),
              }
          )

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          disable_progress_bar: bool = False,
          writer:torch.utils.tensorboard.SummaryWriter = None
          ) -> Dict[str, List]:
  """train and test a pytorch model

  pasees model through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  calculates, prints and stores evaluation metrics throughout.

  Args:
    model: torch.nn.Module
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    epochs: int
    device: torch.device
    disable_progress_bar: bool
    writer: torch.utils.tensorboard.SummaryWriter

  Returns:
    dict of training testing loss, training testing accuracy
    each metric has a value in a list for each epoch.
    in the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    for example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "train_epoch_time": [],
      "test_epoch_time": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs), disable=disable_progress_bar):

      # Perform training step and time it
      train_epoch_start_time = time.time()
      train_loss, train_acc = train_step(epoch=epoch, 
                                        model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device,
                                        disable_progress_bar=disable_progress_bar)
      train_epoch_end_time = time.time()
      train_epoch_time = train_epoch_end_time - train_epoch_start_time
      
      # Perform testing step and time it
      test_epoch_start_time = time.time()
      test_loss, test_acc = test_step(epoch=epoch,
                                      model=model,
                                      dataloader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device,
                                      disable_progress_bar=disable_progress_bar)
      test_epoch_end_time = time.time()
      test_epoch_time = test_epoch_end_time - test_epoch_start_time

      # print what happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"train_epoch_time: {train_epoch_time:.4f} | "
          f"test_epoch_time: {test_epoch_time:.4f}"
      )

      # update results dict
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["train_epoch_time"].append(train_epoch_time)
      results["test_epoch_time"].append(test_epoch_time)
      
      if writer:
          # add results to summarywriter
        writer.add_scalars(main_tag="Loss", 
                              tag_scalar_dict={"train_loss": train_loss,
                                              "test_loss": test_loss},
                              global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", 
                              tag_scalar_dict={"train_acc": train_acc,
                                              "test_acc": test_acc}, 
                              global_step=epoch)

          # close the writer
        writer.close()
      else:
        pass

  # return the filled results at the end of the epochs
  return results