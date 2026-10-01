from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch import nn
from torch.optim import SGD
import torch

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

full_dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

split_generator = torch.Generator().manual_seed(seed)

train_dataset, val_dataset = random_split(
    full_dataset,
    [54000, 6000],
    generator=split_generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

images, labels = next(iter(train_loader))

print("Batch images shape:", images.shape)
print("Batch labels shape:", labels.shape)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

epochs = 3

for epoch in range(epochs):
    model.train()

    loss_sum = 0.0
    sample_count = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

        loss.backward()

        optimizer.step()

    average_loss = loss_sum / sample_count
    print(f"Epoch {epoch + 1}: average training loss = {average_loss:.4f}")

model.eval()

val_correct = 0
val_count = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        val_correct += (predictions == labels).sum().item()
        val_count += labels.size(0)

val_accuracy = val_correct / val_count
print("Validation samples:", val_count)
print(f"Validation accuracy: {val_accuracy:.2%}")