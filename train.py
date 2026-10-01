import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch import nn
from torch.optim import SGD

seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Seed:", seed)

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

train_generator = torch.Generator().manual_seed(seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    generator=train_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

model_name = "mlp"

if model_name == "softmax":
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
    )
elif model_name == "mlp":
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
else:
    raise ValueError(f"Unknown model: {model_name}")

print("Model:", model_name)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

epochs = 3

train_generator.manual_seed(seed)

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

class_correct = torch.zeros(10, dtype=torch.long)
class_count = torch.zeros(10, dtype=torch.long)

confusion_matrix = torch.zeros((10, 10), dtype=torch.long)

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        val_correct += (predictions == labels).sum().item()
        val_count += labels.size(0)

        for class_id in range(10):
            mask = labels == class_id
            class_count[class_id] += mask.sum().item()
            class_correct[class_id] += ((predictions == labels) & mask).sum().item()

        true_labels = labels.cpu().tolist()
        predicted_labels = predictions.cpu().tolist()

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label, predicted_label] += 1

val_accuracy = val_correct / val_count
print("Validation samples:", val_count)
print("Validation correct:", val_correct)
print(f"Validation accuracy: {val_accuracy:.2%}")

for class_id, class_name in enumerate(full_dataset.classes):
    correct = class_correct[class_id].item()
    total = class_count[class_id].item()

    if total == 0:
        print(f"{class_name}: no validation samples")
        continue

    accuracy = correct / total
    print(f"{class_name}: {correct}/{total}, accuracy = {accuracy:.2%}")

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix)

print("Matrix samples:", confusion_matrix.sum().item())
print("Matrix correct:", confusion_matrix.diag().sum().item())
print("Row counts match:", torch.equal(confusion_matrix.sum(dim=1), class_count))
print("Diagonal counts match:", torch.equal(confusion_matrix.diag(), class_correct))