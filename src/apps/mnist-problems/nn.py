from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import csv
import numpy as np
import datetime
import random
import sys
import os
import gc
import re
import logging
import argparse
import pickle
import math

from torchvision.datasets import MNIST
from torchvision import models


current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../', 'utils')
sys.path.append(os.path.abspath(parent_dir))
import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[ logging.StreamHandler()  ])

same_NN_as_DeepSoftLog = load_config.config["same_NN_as_DeepSoftLog"]




class Net(nn.Module):
    def __init__(self,OUT_FEATURES = 10):
        super(Net, self).__init__()

        if same_NN_as_DeepSoftLog is True:
            # https://github.com/jjcmoon/DeepSoftLog/blob/main/deepsoftlog/embeddings/nn_models.py
            
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 6, 5),  # 1 x 28 x 28 -> 6 x 24 x 24
                nn.MaxPool2d(2, 2),  # 6 x 24 x 24 -> 6 x 12 x 12
                nn.ReLU(True),
                nn.Conv2d(6, 16, 5),  # 6 x 12 x 12 -> 16 x 8 x 8
                nn.MaxPool2d(2, 2),  # 16 x 8 x 8 -> 16 x 4 x 4
                nn.ReLU(True),
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, OUT_FEATURES),
            )
        else:

            # Layer 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same')
            self.relu1 = nn.ReLU()
            self.l2_1 = nn.BatchNorm2d(32)

            # Layer 2
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = nn.Dropout(0.25)

            # Layer 3
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
            self.relu2 = nn.ReLU()
            self.l2_2 = nn.BatchNorm2d(64)

            # Layer 4
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout2 = nn.Dropout(0.25)

            self.flatten = nn.Flatten()

            # Layer 6
            self.fc1 = nn.Linear(64 * 7 * 7, 256, bias=False)
            self.bn3 = nn.BatchNorm1d(256)

            # Layer 8
            self.fc2 = nn.Linear(256, 128, bias=False)
            self.bn4 = nn.BatchNorm1d(128)

            # Layer 10
            self.fc3 = nn.Linear(128, 84, bias=False)
            self.bn5 = nn.BatchNorm1d(84)
            self.dropout3 = nn.Dropout(0.25)

            # Output
            self.out = nn.Linear(84, OUT_FEATURES)

    def forward(self, x):
        if same_NN_as_DeepSoftLog is True:
            x = self.encoder(x)
            x = x.view(-1, 16 * 4 * 4)  # Flatten
            x = self.classifier(x)
            return x
        else:
            x = self.relu1(self.l2_1(self.conv1(x)))
            x = self.bn1(self.conv2(x))
            x = self.pool1(F.relu(x))
            x = self.dropout1(x)

            x = self.relu2(self.l2_2(self.conv3(x)))
            x = self.bn2(self.conv4(x))
            x = self.pool2(F.relu(x))
            x = self.dropout2(x)

            x = self.flatten(x)
            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))
            x = self.dropout3(F.relu(self.bn5(self.fc3(x))))

            x = self.out(x)
            return x


class MNISTDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list of tuples): A list where each tuple contains:
                - image_data: a flat list of pixels for a 28x28 MNIST image
                - truth_label: the integer label for the image
                - puzzle_id
                - row_id 
                - col_id
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_data, truth_label, puzzle_id, row_id, col_id = self.data_list[idx]

        # Assuming image_data is a flat list of 784 pixels (28x28 image)
        image_tensor = torch.tensor(image_data, dtype=torch.float).view(28, 28)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label_tensor = torch.tensor(truth_label, dtype=torch.long)

        return image_tensor, label_tensor
                


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train() 
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
                
        total_loss += loss.item()
        optimizer.step()  

        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    average_loss = total_loss / len(train_loader)
    logging.info(f'Epoch {epoch} Average Training Loss: {average_loss:.4f}')


def test(model, device, test_loader, set_name="Test set"):
    model.eval()  
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            pred = output.argmax(dim=1)
                
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)

    average_loss = total_loss / total_samples
    accuracy_test = 100.0 * total_correct / total_samples
    logging.info('\n {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set_name, average_loss, total_correct, total_samples, accuracy_test))
    return accuracy_test



def export(model, device, dataset, indexes, file, OUT_FEATURES = 10, fromRAW = False):
    model.eval()
    l = []
    for i, image in enumerate(dataset):
        sample_image, label = dataset[i]
        image_reshaped = sample_image.unsqueeze(0).float()

        with torch.no_grad():
            sample_image = sample_image.unsqueeze(0).float().to(device)  
            
            
            output = model(sample_image)

            
            softmax = torch.nn.functional.softmax(output, dim=1)
            probabilities = softmax.cpu().numpy().flatten().tolist()
            predicted_label = torch.argmax(softmax, dim=1).item()
            
        # constructing the row of the CSV: 
        # id, label (ground thruth), probabilities of label 0,1,2,3...,OUT_FEATURES-1, 
        # predicted label, from train or not (only for mnist sudoku), original_id
        current_row = [i]
        if fromRAW:
            current_row.append(label.item())  
        else:
            current_row.append(label)  

        current_row.extend(probabilities) 
        
        current_row.append(predicted_label)


        if fromRAW is False:
            current_row.append(1 if indexes[i][1] == 1 else 0)
            current_row.append(indexes[i][0])
        else:
            current_row.append(0)
            current_row.append(i)

        l.append(current_row)



    fieldsnames = ['id','label']
    for x in range(0,OUT_FEATURES):
        fieldsnames.append(str(x))
    fieldsnames.append("predicted_label")
    fieldsnames.append("train")
    fieldsnames.append("original_id")

    with open(file,'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldsnames)
        writer.writeheader()
        for x in l:
            d = {}
            for a,b in zip(fieldsnames,x):
                d[a] =b
            writer.writerow(d)


def prepare_data(dataset_trainvalid, dataset_test, split_size_train, split_size_valid, split_size_test):
    total_indices_train = torch.arange(len(dataset_trainvalid))
    set_total_indices = set(total_indices_train.tolist())

    indices_train = torch.tensor(random.sample(range(len(dataset_trainvalid)), split_size_train))
    indices_train = indices_train.clone().detach()

    
    if same_NN_as_DeepSoftLog or split_size_train == 60000:
        # train and val overlaps: DeepSoftLog config
        indices_val = torch.tensor(random.sample(range(len(dataset_trainvalid)), min(split_size_valid, len(dataset_trainvalid))))
    else:
        # train and val are disjoints
        set_indices_train = set(indices_train.tolist())
        set_indices_val = set_total_indices - set_indices_train
        list_indices_val = list(set_indices_val)
        # by default: set split_size_valid = int(split_size_train/5)
        indices_val = torch.tensor([list_indices_val[x] for x in torch.tensor(random.sample(range(len(set_indices_val)), min(split_size_valid, len(set_indices_val))))])


    indices_test = torch.tensor(random.sample(range(len(dataset_test)), split_size_test))
    indices_test = indices_test.clone().detach()

    trainset_1 = torch.utils.data.Subset(dataset_trainvalid, indices_train)
    validationset_1 = torch.utils.data.Subset(dataset_trainvalid, indices_val)
    testset_1 = torch.utils.data.Subset(dataset_test, indices_test)

    trainset_indexes = [(x.item(),1) for x in indices_train]
    validationset_indexes = [(x.item(),1) for x in indices_val]
    testset_indexes = [(x.item(),2) for x in indices_test]

    logging.info("Train set size: "+str(len(trainset_1)))
    logging.info("Validation set size: "+ str(len(validationset_1)))
    logging.info("Test set size: "+str(len(testset_1)))
    
    return trainset_1, validationset_1, testset_1, trainset_indexes, validationset_indexes, testset_indexes

def MakeDataFromRaw(all_train_data, all_valid_data, all_test_data):
    # transform from raw image
    transform_from_raw = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_raw_data = all_train_data
    valid_raw_data = all_valid_data
    test_raw_data = all_test_data

    dataset_train = MNISTDataset(train_raw_data, transform=transform_from_raw)
    dataset_valid = MNISTDataset(valid_raw_data, transform=transform_from_raw)
    dataset_test = MNISTDataset(test_raw_data, transform=transform_from_raw)

    trainset_indexes = [i for i in range(0,len(all_train_data))]
    validationset_indexes = [i for i in range(0, len(all_valid_data))]
    testset_indexes = [i for i in  range(0, len(all_test_data))]
    

    logging.info("Train set size: "+ str(len(dataset_train)))
    logging.info("Validation set size:"+ str(len(dataset_valid)))
    logging.info("Test set size: "+ str(len(dataset_test)))
    
    return dataset_train, dataset_valid, dataset_test, trainset_indexes, validationset_indexes, testset_indexes




def run_mnist_experiment(split_size_train,
                        split_size_valid,
                        split_size_test, 
                        out_features = 10,
                        all_train_data = [], 
                        all_valid_data = [], 
                        all_test_data = []):

    logging.info(f"problem studied: {load_config.config['problem_studied']}")

    if same_NN_as_DeepSoftLog is False:
        logging.info("Using Pi-NeSy's CNN.")
    else:
        logging.info("Using DeepSoftLog's CNN for mnist-add.")
    
    
    transform_basic = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # train+valid
    dataset_trainvalid = datasets.MNIST(load_config.config["MNIST_data_directory"], train=True, download=True,
                            transform=transform_basic)
    # test
    dataset_test = datasets.MNIST(load_config.config["MNIST_data_directory"], train=False, download=True,
                            transform=transform_basic)


    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    

    # by default: use the hyperparameters of Pi-NeSy's CNN from config.json 
    if load_config.config['problem_studied'].startswith("addition"):
        batch_size = load_config.config["PiNeSy_CNN_batch_size_for_mnistadd"]
    elif load_config.config['problem_studied'].startswith("sudoku_4"):
        batch_size = load_config.config["PiNeSy_CNN_batch_size_for_mnistsudoku4x4"]
    elif load_config.config['problem_studied'].startswith("sudoku_9"):
        batch_size = load_config.config["PiNeSy_CNN_batch_size_for_mnistsudoku9x9"]

    test_batch_size = load_config.config["PiNeSy_CNN_test_batch_size"]
    lr = load_config.config["PiNeSy_CNN_lr"]
    gamma = load_config.config["PiNeSy_CNN_gamma"]
    epochs = load_config.config["PiNeSy_CNN_epochs"]
    log_interval = 10

    # if using DeepSoftLog's CNN for mnist-add, we change some hyperparameters 
    # (and the CNN architecture is LeNet5, see beginning of file)
    if same_NN_as_DeepSoftLog is True:
        if load_config.config['problem_studied'].startswith("addition"):
            epochs = load_config.config["DeepSoftLog_mnistadd_epochs"]
            batch_size = load_config.config["DeepSoftLog_mnistadd_batch_size"]
            test_batch_size = 1000
        else:
            logging.info("No other problem were studied using DeepSoftlog's CNN.")
            sys.exit(0)



    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.info("use_mps:" + str(use_mps))
    logging.info("same_NN_as_DeepSoftLog:" + str(same_NN_as_DeepSoftLog))
    logging.info("PyTorch version:" + str(torch.__version__))

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    if use_mps:
        train_kwargs = {'batch_size': batch_size, 'num_workers': 0, 'shuffle': True, 'pin_memory':False}
        test_kwargs = {'batch_size': test_batch_size, 'num_workers': 0, 'shuffle': True, 'pin_memory':False}

    
    # in the case of mnist addition, we only load the function run_mnist_experiment with three parameters: 
    # split_size_train: size of train, split_size_valid: size of valid, split_size_test: size of test
    # in the case of mnist sudoku, we load the function run_mnist_experiment with the following parameters: 
    # 0, 0, 0, out_features = 4/10,  all_{train|valid|test}_data \neq []: data from vspc

    if load_config.config['problem_studied'].startswith("addition"): # the problem is mnist addition: generating the train/test/validation sets
        trainset_1, validationset_1, testset_1, trainset_indexes, validationset_indexes, testset_indexes = prepare_data(dataset_trainvalid, dataset_test, split_size_train, split_size_valid, split_size_test)
    else: # the problem is mnist sudoku : using data of vspc for generating the train/test/validation sets
        trainset_1, validationset_1, testset_1, trainset_indexes, validationset_indexes, testset_indexes = MakeDataFromRaw(all_train_data, all_valid_data, all_test_data)

    train_loader = torch.utils.data.DataLoader(trainset_1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset_1, **test_kwargs)
    valid_loader = torch.utils.data.DataLoader(validationset_1, **test_kwargs)



    model = Net(out_features).to(device)

    # configuration of the optimizer when we use DeepSoftLog's CNN
    if same_NN_as_DeepSoftLog is True:
        if load_config.config['problem_studied'].startswith("addition"):
            optimizer = optim.AdamW(model.parameters(), lr=load_config.config["DeepSoftLog_mnistadd_lr"], weight_decay=load_config.config["DeepSoftLog_mnistadd_weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=load_config.config["DeepSoftLog_mnistadd_optimizerTmax"], eta_min=load_config.config["DeepSoftLog_mnistadd_optimizerEtaMin"])
    # by default: Pi-NeSy's CNN uses Adadelta and StepLR
    else: 
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    # We train if we use PiNeSy CNN or, DeepSoftLog CNN was not trained for mnist-addition-k problems
    if (same_NN_as_DeepSoftLog is False) or\
        (same_NN_as_DeepSoftLog is True and load_config.config['problem_studied'].startswith("addition") and load_config.config["DeepSoftLog_checkpointmnistaddfile"] ==  ""):
        # training process
        for epoch in range(1, epochs + 1):
            train(log_interval, model, device, train_loader, optimizer, epoch)
            accuracy_val = test(model, device, valid_loader, "Validation set (subset of MNIST train dataset)")
            accuracy_test = test(model, device, test_loader, "Test set")
    
    
    if same_NN_as_DeepSoftLog and load_config.config['problem_studied'].startswith("addition") or split_size_train == 60000:
        if same_NN_as_DeepSoftLog and load_config.config["DeepSoftLog_checkpointmnistaddfile"] == "":

            validloader_file = load_config.config["DeepSoftLog_mnistadd_validloader_file"]

            if validloader_file == "":
                # specific validation dataset, which is a subset of the training dataset whose images are slightly modified
                validation_transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=(-2, 2),      
                        translate=(0.02, 0.02),      
                        scale=(0.98, 1.02),
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                dataset_trainvalid_modified = datasets.MNIST(load_config.config["MNIST_data_directory"], train=True, download=True,
                                                            transform=validation_transform)
                validation_indices = [idx for idx, flag in validationset_indexes]
                validationset_1 = torch.utils.data.Subset(dataset_trainvalid_modified, validation_indices)
            
                valid_loader = torch.utils.data.DataLoader(validationset_1, **test_kwargs)
                
                accuracy_val = test(model, device, valid_loader, "Validation set  (subset of MNIST train dataset)  with slight rotations, shift, scaling performed on samples")

                validation_data = torch.stack([dataset_trainvalid_modified[i][0] for i in validation_indices])
                validation_labels = torch.tensor([dataset_trainvalid_modified[i][1] for i in validation_indices])

                save_path = f"./tmp/validdataset-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
                torch.save({"data": validation_data, "labels": validation_labels}, save_path)

                
                logging.info(f"Validation DataLoader (with slight rotations, shift, scaling) saved to {save_path}")

                load_config.update_config("DeepSoftLog_mnistadd_validloader_file", save_path)
            else:
                loaded_data = torch.load(validloader_file)

                validation_data = loaded_data["data"]
                validation_labels = loaded_data["labels"]

                validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)
                valid_loader = torch.utils.data.DataLoader(validation_dataset, **test_kwargs)

                logging.info(f"Validation DataLoader (with slight rotations, shift, scaling)  {validloader_file} loaded from config.json")

                accuracy_val = test(model, device, valid_loader, "Validation set  (subset of MNIST train dataset)  with slight rotations, shift, scaling performed on samples")
    else:
        accuracy_val = test(model, device, valid_loader, "Validation set (subset of MNIST train dataset)")
    
    # saving checkpoint if we have performed training, or we load a checkpoint if we want to use the trained DeepSoftLog CNN for mnist-add
    if (same_NN_as_DeepSoftLog is False) or\
        (same_NN_as_DeepSoftLog is True and load_config.config['problem_studied'].startswith("addition") and load_config.config["DeepSoftLog_checkpointmnistaddfile"] ==  ""):
        checkpoint_filename = f"./tmp/checkpoint_{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}_{random.randint(0, 9999):04d}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info("saved checkpoint of the model in: " + checkpoint_filename)
    else:
        if same_NN_as_DeepSoftLog is True and load_config.config['problem_studied'].startswith("addition") and load_config.config["DeepSoftLog_checkpointmnistaddfile"] != "":
            model.load_state_dict(torch.load(load_config.config["DeepSoftLog_checkpointmnistaddfile"]))
            logging.info("Loaded checkpoint for MNIST addition-k with DeepSoftLog's CNN learned model:" + load_config.config["DeepSoftLog_checkpointmnistaddfile"])
            checkpoint_filename = load_config.config["DeepSoftLog_checkpointmnistaddfile"]
        
        accuracy_val = test(model, device, valid_loader, "Validation set")
        accuracy_test = test(model, device, test_loader, "Test set")
    

    logging.info("Begin export....")

    filetrain = "./tmp/train" + str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + "_" + str(random.randint(0, 9999)) +  str(".csv")
    filevalidation = "./tmp/validation" + str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + "_" + str(random.randint(0, 9999)) + str(".csv")
    filetest = "./tmp/test" + str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + "_" + str(random.randint(0, 9999)) + str(".csv")

    if load_config.config['problem_studied'].startswith("addition"): #  in the case that the problem is mnist addition
        fromRaw = False
    else: # in the case that the problem is mnist sudoku: we use vspc raw data
        fromRaw = True
    
    logging.info(f"exporting from trainset (size:{len(trainset_1)}) to {filetrain}")
    export(model, device, trainset_1, trainset_indexes, filetrain, out_features, fromRaw)

    logging.info(f"exporting from validationset (size:{len(validationset_1)}) to {filevalidation}")
    export(model, device, validationset_1, validationset_indexes, filevalidation, out_features, fromRaw)

    logging.info(f"exporting from testset (size:{len(testset_1)}) to {filetest}")
    export(model, device, testset_1, testset_indexes, filetest, out_features, fromRaw)

    logging.info("Export done.")
    
    
    del dataset_trainvalid, dataset_test
    gc.collect()
    
    del model, optimizer, scheduler, train_loader, test_loader
    gc.collect()

    if 'all_train_data' in locals(): del all_train_data
    if 'all_valid_data' in locals(): del all_valid_data
    if 'all_test_data' in locals(): del all_test_data
    gc.collect()

    if use_cuda:
        torch.cuda.empty_cache()
    elif use_mps:
        torch.mps.empty_cache()

    return filetrain, filevalidation, filetest, accuracy_test, accuracy_val, checkpoint_filename