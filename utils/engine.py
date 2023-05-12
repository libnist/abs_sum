# Import libraries
import torch

from tqdm.auto import trange

from types import FunctionType

from collections import defaultdict

import pathlib

from .save_and_load import save_model

import wandb

# Train step: Performs one epoch on the training data


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function: FunctionType,
               optimizer: torch.optim.Optimizer,
               device: str,
               log_per_batch: int = 100):
    # Putting the model in train mode. Activates things like dropout layers.
    model.train()

    # Initializing loss and accuracy values
    train_loss, train_acc = 0, 0

    # Get the number of batches in order to avoid recalculating it each
    # time we want it again.
    num_batches = len(dataloader)

    t_range = trange(num_batches)
    t_range.set_description("\t\tTrain Batches")

    # The next step is to train the model with our dataloader.
    for batch, (X, y, z) in zip(t_range, dataloader):
        # Device of the input tensors are already set in the
        # dataloader initialization.

        batch += 1

        # Getting data in the form of the models input

        # Preparing inputs
        inputs = (X.to(device), y.to(device))

        z = z.to(device)

        # Forward pass
        # The output is in shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        logits = model(*inputs)

        # Calculating loss
        loss = loss_function(logits.reshape(-1, logits.shape[-1]),
                             z.reshape(-1))
        train_loss += loss.item()

        # Backward pass and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating the accuracy of the model

        # [batch_size, summary_token_length,]
        preds = logits.softmax(dim=-1).argmax(dim=-1)

        accuracy = accuracy_function(preds, z)
        train_acc += accuracy.item()

        if ((not batch % log_per_batch)
                or (batch == num_batches)):
            loss_to_print = train_loss / batch
            accuracy_to_print = train_acc / batch

            t_range.set_postfix(Train_Loss=f"{loss_to_print:8.4f}",
                                Train_Accuracy=f"{accuracy_to_print*100:4.2f}")
    train_loss /= num_batches
    train_acc /= num_batches
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              accuracy_function: FunctionType,
              device: str):
    # Putting the model in eval mode
    model.eval()
    eval_loss, eval_acc = 0, 0

    num_batches = len(dataloader)
    t_range = trange(num_batches)
    t_range.set_description("\t\tTest Batches")

    with torch.inference_mode():
        for batch, (X, y, z) in zip(t_range, dataloader):
            batch += 1
            # Getting the inputs in proper shape
            # Preparing inputs
            inputs = (X.to(device), y.to(device))
            z = z.to(device)

            # Forward pass
            # The output is in shape:
            # [batch_size, summary_token_length, sum_vocab_size]
            logits = model(*inputs)

            # Calculating loss
            loss = loss_function(logits.reshape(-1, logits.shape[-1]),
                                 z.reshape(-1))
            eval_loss += loss.item()

            # [batch_size, summary_token_length]
            preds = logits.softmax(dim=-1).argmax(dim=-1)

            accuracy = accuracy_function(preds, z)
            eval_acc += accuracy.item()
            
            if batch == num_batches:
                eval_loss /= num_batches
                eval_acc /= num_batches

                t_range.set_postfix(Test_Loss=f"{eval_loss:8.4f}",
                                    Test_Accuracy=f"{eval_acc*100:4.2f}")
    return eval_loss, eval_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          accuracy_function: FunctionType,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str,
          initial_epoch: int = None,
          val_dataloader: torch.utils.data.DataLoader = None,
          lr_scheduler=None,
          path: pathlib.Path = None,
          model_name: str = None,
          log_per_batch: int = 100,
          wandb_config: dict = None,
          wandb_proj: str = None,
          wandb_id: str = None,
          tb_writer: torch.utils.tensorboard.SummaryWriter = None):
    # If a path is defined in order for the model to be saved
    # a model_name also should be defined, it will be used to save
    # the model.
    if path:
        assert model_name, "Define model_name parameter."

    # If wandb_config is defined, will be initializing it, so we can
    # make logs in Weights and Biases.
    if wandb_config:
        assert wandb_proj, "Define wandb_proj as the project name."
        if wandb_id:
            wandb.init(id=wandb_id,
                       project=wandb_proj,
                       config=wandb_config,
                       name=model_name,
                       resume="must")
        else:
            wandb.init(project=wandb_proj,
                       config=wandb_config,
                       name=model_name)

    # The dictionary below will containg all the loss and accuracy
    # reports from the training proccess
    results = defaultdict(list)

    # Putting our model into the predefined device
    model.to(device)

    if initial_epoch:
        range_iter = range(initial_epoch, epochs)
    else:
        range_iter = range(epochs)

    # Iterating as many epochs we need and updating our model weights.
    for epoch in range_iter:
        epoch += 1
        print(f"{'*'*20} Start of Epoch {epoch}/{epochs} {'*'*20}", end="")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           accuracy_function=accuracy_function,
                                           optimizer=optimizer,
                                           log_per_batch=log_per_batch,
                                           device=device)

        # Append the results of the current finished epoch.
        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_acc)

        # We'll be evaluate our model in case of having an validation
        # dataset.
        if val_dataloader:
            test_loss, test_acc = test_step(model=model,
                                            dataloader=val_dataloader,
                                            loss_function=loss_function,
                                            accuracy_function=accuracy_function,
                                            device=device)
            # Append the results of the current finished validation epoch.
            results["val_losses"].append(test_loss)
            results["val_accuracies"].append(test_acc)

        # If there is a learning rate scheduler defined, after each train_step
        # will call it's .step() in order to update our optimizers lr.
        if lr_scheduler:
            lr_scheduler.step()

        # Save the model in case of having a path to save it.
        if path:
            save_model(model=model,
                       path=path,
                       name=model_name,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)

        # Report our results to wandb
        if wandb_config:
            log = {"train_loss": train_loss,
                   "train_accuracy": train_acc}
            if val_dataloader:
                log.update({"val_loss": test_loss,
                            "val_accuracy": test_acc})
            wandb.log(log, step=epoch, commit=True)
            print(f"\tWandb Logs are reported!")

        # Report our results to tensorboard
        if tb_writer:
            acc_log = {"train": train_acc}
            loss_log = {"train": train_loss}
            if val_dataloader:
                acc_log.update({"val": test_acc})
                loss_log.update({"val": test_loss})
            tb_writer.add_scalars(main_tag="Loss",
                                  tag_scalar_dict=loss_log,
                                  global_step=epoch)
            tb_writer.add_scalars(main_tag="Accuracy",
                                  tag_scalar_dict=acc_log,
                                  global_step=epoch)
            print(f"\tTensorBoard Logs are reported!")
        print(f"{'*'*20} End of Epoch {epoch}{'*'*20}\n\n")

    if wandb_config:
        wandb.finish()
    if tb_writer:
        tb_writer.close()

    return results
