import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import VGG
matplotlib.style.use('ggplot')


class Trainer():
    def __init__(self, epochs, batch_size, vgg_config):
        self.epochs = epochs
        self.batch_size = batch_size
        self.vgg_config = vgg_config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_data()
        self.load_model()

    def prepare_data(self):
        train_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5), std=(0.5))])
        valid_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5), std=(0.5))])

        # training dataset and data loader
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True,
            download=True,
            transform=train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True)
        # validation dataset and dataloader
        valid_dataset = torchvision.datasets.MNIST(
            root='./data', train=False,
            download=True,
            transform=valid_transform)
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False)

    def load_model(self):
        # instantiate the model
        self.model = VGG(
            config=self.vgg_config,
            in_channels=1,
            num_classes=10).to(self.device)
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO]: {total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[INFO]: {total_trainable_params:,} trainable parameters.")
        # the loss function
        self.criterion = nn.CrossEntropyLoss()
        # the optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01, momentum=0.9,
            weight_decay=0.0005)

    def train(self):
        self.model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(self.train_dataloader),
                            total=len(self.train_dataloader)):
            counter += 1

            image, labels = data
            image = image.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(image)
            # calculate the loss
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            self.optimizer.step()

        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct /
                            len(self.train_dataloader.dataset))
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()

        # we need two lists to keep track of class-wise accuracy
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        print('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.valid_dataloader),
                                total=len(self.valid_dataloader)):
                counter += 1

                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = self.model(image)
                # calculate the loss
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
                # calculate the accuracy for each class
                correct = (preds == labels).squeeze()
                for i in range(len(preds)):
                    label = labels[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct /
                            len(self.valid_dataloader.dataset))
        # print the accuracy for each class after evey epoch
        # the values should increase as the training goes on
        print('\n')
        for i in range(10):
            print(f"Accuracy of {i}: {100*class_correct[i]/class_total[i]}")
        return epoch_loss, epoch_acc

    def run(self):
        torch.cuda.empty_cache()
        # start the training
        # lists to keep track of losses and accuracies
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        for epoch in range(self.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc = self.train()
            valid_epoch_loss, valid_epoch_acc = self.validate()
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print('\n')
            print(f"Train loss: {train_epoch_loss:.3f}, train acc: {train_epoch_acc:.3f}")
            print(f"Valid loss: {valid_epoch_loss:.3f}, valid acc: {valid_epoch_acc:.3f}")
            print('-'*50)

        # save the trained model to disk
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, '../outputs/model.pth')
        # accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-',
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-',
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('../outputs/accuracy.jpg')
        plt.show()
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-',
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-',
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('../outputs/loss.jpg')
        plt.show()

        print('TRAINING COMPLETE')
