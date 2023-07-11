from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        output = self.fc2(x) 
        return output

def main():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    training_dataloader = DataLoader(train_dset, shuffle=True, batch_size=64)
    validation_datalaoder = DataLoader(test_dset, shuffle=False, batch_size=64)

    model = BasicNet()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    accelerator = Accelerator()

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )

    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for batch in training_dataloader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
    scheduler.step()

    validation_datalaoder = accelerator.prepare(validation_datalaoder)

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in validation_datalaoder:
            inputs, targets = batch
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            # Gathers tensor and potentially drops duplicates in the last batch if on a distributed system.
            all_preds, all_targets = accelerator.gather_for_metrics((pred, targets))
            if accelerator.is_main_process:
                correct += all_preds.eq(all_targets.view_as(all_preds)).sum().item()
    if accelerator.is_main_process:            
        accelerator.print(f'Accuracy: {100. * correct / len(validation_datalaoder.dataset)}')

    

if __name__ == "__main__":
    main()

# RUN: CUDA_VISIBLE_DEVICES="6,7" accelerate launch --config_file ./my_config_file.yaml Accelerate_example.py