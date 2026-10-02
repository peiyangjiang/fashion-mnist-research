import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch import nn
from torch.optim import SGD

class RelabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, self.labels[index]

init_seed = 42
split_seed = 42
shuffle_seed = 42

torch.manual_seed(init_seed)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(f"Seeds: init={init_seed}, split={split_seed}, shuffle={shuffle_seed}")

full_dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

split_generator = torch.Generator().manual_seed(split_seed)

train_pool, val_dataset = random_split(
    full_dataset,
    [54000, 6000],
    generator=split_generator
)

train_size = 18000
subset_seed = 42

subset_generator = torch.Generator().manual_seed(subset_seed)
subset_indices = torch.randperm(
    len(train_pool),
    generator=subset_generator,
).tolist()

train_dataset = Subset(train_pool, subset_indices[:train_size])

train_pool_indices = torch.tensor(train_pool.indices, dtype=torch.long)
train_indices = train_pool_indices[train_dataset.indices]

original_labels = full_dataset.targets[train_indices].clone()
noisy_labels = original_labels.clone()

original_val_labels = full_dataset.targets[val_dataset.indices].clone()

noise_rate = 0.2
noise_seed = 42
print(f"Noise rate: {noise_rate:.0%}")
print("Noise seed:", noise_seed)

noise_generator = torch.Generator().manual_seed(noise_seed)

num_noisy = int(noise_rate * len(train_dataset))
noisy_indices = torch.randperm(
    len(train_dataset),
    generator=noise_generator,
)[:num_noisy]

label_offsets = torch.randint(
    1,
    10,
    (num_noisy,),
    generator=noise_generator,
)

noisy_labels[noisy_indices] = (
    original_labels[noisy_indices] + label_offsets
) % 10

noisy_train_dataset = RelabeledDataset(train_dataset, noisy_labels)

actual_noisy = (noisy_labels != original_labels).sum().item()
val_unchanged = torch.equal(
    full_dataset.targets[val_dataset.indices],
    original_val_labels,
)

print("Changed training labels:", actual_noisy)
print("Validation labels unchanged:", val_unchanged)

assert actual_noisy == num_noisy
assert val_unchanged

train_generator = torch.Generator().manual_seed(shuffle_seed)

train_loader = DataLoader(
    noisy_train_dataset,
    batch_size=32,
    shuffle=True,
    generator=train_generator,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

noisy_eval_dataset = Subset(train_dataset, noisy_indices.tolist())

noisy_eval_loader = DataLoader(
    noisy_eval_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
)

noisy_eval_labels = noisy_labels[noisy_indices]

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

max_steps = 5000
step = 0

if len(train_loader) == 0:
    raise ValueError("Training data must contain at least one full batch.")

print("Train size:", len(train_dataset))
print("Subset seed:", subset_seed)
print("Max training steps:", max_steps)

train_generator.manual_seed(shuffle_seed)

while step < max_steps:
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

        step += 1
        if step >= max_steps:
            break

    average_loss = loss_sum / sample_count
    print(f"Step {step}: average loss for this pass = {average_loss:.4f}")

print("Completed training steps:", step)

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

noise_fit_correct = 0
noise_clean_correct = 0
noise_eval_count = 0

model.eval()

with torch.no_grad():
    for images, clean_labels in noisy_eval_loader:
        images = images.to(device)
        clean_labels = clean_labels.to(device)

        batch_size = images.size(0)
        noisy_targets = noisy_eval_labels[
            noise_eval_count : noise_eval_count + batch_size
        ].to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        noise_fit_correct += (predictions == noisy_targets).sum().item()
        noise_clean_correct += (predictions == clean_labels).sum().item()
        noise_eval_count += batch_size

assert noise_eval_count == num_noisy
assert noise_fit_correct + noise_clean_correct <= noise_eval_count

print("\nCorrupted training samples evaluated:", noise_eval_count)

if noise_eval_count > 0:
    print("Wrong-label matches:", noise_fit_correct)
    print(f"Wrong-label fit rate: {noise_fit_correct / noise_eval_count:.2%}")
    print("Original-label matches:", noise_clean_correct)
    print(f"Original-label accuracy: {noise_clean_correct / noise_eval_count:.2%}")
else:
    print("Wrong-label fit rate: N/A (no corrupted training samples)")
    print("Original-label accuracy: N/A (no corrupted training samples)")
