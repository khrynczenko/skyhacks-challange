import os
import time
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import softmax


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                device='cpu', artifacts_directory="artifacts/"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            results = []
            targets = []
            correct_rows = []
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], 
                                       total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs > 0.5
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                correct_rows.append(np.all(np.asarray(preds.detach().cpu().numpy(), dtype=np.uint8) == labels.detach().cpu().numpy()))
                running_corrects += np.sum(np.asarray(preds.detach().cpu().numpy(), dtype=np.uint8) == labels.detach().cpu().numpy())
                results.append(np.asarray(preds.detach().cpu().numpy().flatten(), dtype=np.uint8))
                targets.append(labels.detach().cpu().numpy().flatten())
                pass

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / (len(dataloaders[phase]) * labels.shape[1])
            correct_rows = np.sum(correct_rows)

            print('{} Loss: {:.4f} Acc: {:.4f} Correct rows {}'.format(
                phase, epoch_loss, epoch_acc, correct_rows))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 
                       os.path.join(artifacts_directory,
                                    model.__class__.__name__ + f"-e{epoch}.pt"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_2(model, dataloaders, criterion, optimizer, num_epochs=25,
                  device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs)
                    labels = labels.long()[0, :]
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += (preds.int() == labels.int())
                pass

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / (len(dataloaders[phase]))
            epoch_acc = epoch_acc.detach().cpu().numpy()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc[0]))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), os.path.join("./artifacts", f"{epoch}.pt"))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_3(model, dataloaders, criterion, optimizer, num_epochs=25,
                device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            accuracy1 = 0.0
            accuracy2 = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                labels = labels.type(torch.LongTensor)
                labels = labels - 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1, outputs2 = model(inputs)
                    loss1 = criterion(outputs1, labels[:, 0])
                    loss2 = criterion(outputs2, labels[:, 1])
                    loss = 0.5 * loss1 + 0.5 * loss2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                accuracy1 += (torch.argmax(softmax(outputs1, dim=1), dim=1) == labels[:, 0]).float().sum() / float(labels.shape[0])
                accuracy2 += (torch.argmax(softmax(outputs2, dim=1), dim=1) == labels[:, 1]).float().sum() / \
                            float(labels.shape[0])
                pass

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / (len(dataloaders[phase]) * labels.shape[1])

            print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f}'.format(
                phase, epoch_loss, accuracy1 / float(len(dataloaders[phase])), accuracy2 / float(len(dataloaders[phase]))))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(r"C:\skyhacks\artifacts", f"{epoch}.pt"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
