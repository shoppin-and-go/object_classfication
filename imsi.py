# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# import timm
# from PIL import Image
# import os
# from dataclasses import dataclass
#
#
# @dataclass
# class CNN_config:
#     model_name: str = "SwinTransformerV2"  # 모델 이름
#     train_dir: str = "./train"  # 학습 데이터 경로
#     test_dir: str = "./test"  # 테스트 데이터 경로
#     model_dir: str = "./model"  # 모델 저장 경로
#     mode: str = "train"  # "train" 또는 "test"
#     batch_size: int = 32  # 배치 사이즈
#     learning_rate: float = 0.001  # 학습률
#     epochs: int = 100  # 학습 횟수
#     pretrained: bool = False  # 사전 학습된 모델 사용 여부
#     class_file: str = "./product.txt"  # 클래스 파일 경로
#     image_size: (int, int) = (256, 256)  # 이미지 크기
#     normalize_mean: (float, float, float) = (0.485, 0.456, 0.406)  # 정규화 평균값
#     normalize_std: (float, float, float) = (0.229, 0.224, 0.225)  # 정규화 표준편차
#     num_workers: int = 0  # 데이터 로드에 사용할 CPU 코어 수
#
#     def __post_init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_dir = os.path.join("model", self.model_name)
#
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#
#         with open(self.class_file, "r") as f:
#             self.classes = f.read().splitlines()
#
#         self.num_classes = len(self.classes)
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
#
#
# def read_images_and_labels(config):
#     if config.mode == "train":
#         data_dir = config.train_dir
#     elif config.mode == "test":
#         data_dir = config.test_dir
#     else:
#         raise ValueError("Mode must be 'train' or 'test'.")
#
#     print(f"Reading images and labels from {data_dir} ({config.mode})")
#
#     images = []
#     labels = []
#
#     category = config.classes
#
#     for c in category:
#         path = os.path.join(data_dir, c)
#         if not os.path.exists(path):
#             print(f"Warning: {path} does not exist.")
#             continue
#         for img_name in os.listdir(path):
#             img_path = os.path.join(path, img_name)
#             if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 images.append(img_path)
#                 labels.append(c)
#
#     print(f"Reading images and labels done. {len(images)} images found.")
#     return images, labels
#
#
# def create_model(config):
#     model = timm.create_model('swinv2_tiny_window16_256', pretrained=config.pretrained, num_classes=config.num_classes)
#     return model
#
#
# class ImageDataset(Dataset):
#     def __init__(self, images, labels, config, transform=None):
#         self.images = images
#         self.labels = labels
#         self.class_to_idx = config.class_to_idx
#         self.transform = transform
#         self.image_size = config.image_size
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         label_name = self.labels[idx]
#         label = self.class_to_idx[label_name]
#
#         try:
#             image = Image.open(image_path).convert("RGB")
#             if self.transform:
#                 image = self.transform(image)
#         except Exception as e:
#             print(f"Error loading image {image_path}: {e}")
#             image = torch.zeros(3, self.image_size[0], self.image_size[1])
#
#         return image, label
#
#
# def create_dataloader(config, images, labels):
#     if config.mode == "train":
#         data_transforms = transforms.Compose([
#             transforms.Resize(config.image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(20),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
#         ])
#     else:
#         data_transforms = transforms.Compose([
#             transforms.Resize(config.image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
#         ])
#
#     dataset = ImageDataset(images, labels, config, transform=data_transforms)
#
#     if config.mode == "train":
#         train_size = int(0.8 * len(dataset))
#         val_size = len(dataset) - train_size
#         train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
#         train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
#         val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
#
#         return train_loader, val_loader
#     else:
#         test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
#         return test_loader
#
#
# def train_SwinTransformerV2(config):
#     images, labels = read_images_and_labels(config)
#     train_loader, val_loader = create_dataloader(config, images, labels)
#
#     model = create_model(config)
#     model = model.to(config.device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     best_acc = 0.0
#     best_model_path = None
#
#     print(f"Start training {config.model_name}...")
#     for epoch in range(config.epochs):
#         model.train()
#         train_loss = 0.0
#         train_corrects = 0
#
#         for inputs, labels in train_loader:
#             inputs = inputs.to(config.device)
#             labels = labels.to(config.device)
#
#             optimizer.zero_grad()
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             _, preds = torch.max(outputs, 1)
#             train_loss += loss.item() * inputs.size(0)
#             train_corrects += torch.sum(preds == labels.data)
#
#         train_loss = train_loss / len(train_loader.dataset)
#         train_acc = train_corrects.double() / len(train_loader.dataset)
#
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs = inputs.to(config.device)
#                 labels = labels.to(config.device)
#
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 _, preds = torch.max(outputs, 1)
#                 val_loss += loss.item() * inputs.size(0)
#                 val_corrects += torch.sum(preds == labels.data)
#
#         val_loss = val_loss / len(val_loader.dataset)
#         val_acc = val_corrects.double() / len(val_loader.dataset)
#
#         scheduler.step()
#
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_model_path = os.path.join(config.model_dir, f"{config.model_name}_best.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Best model saved at {best_model_path}")
#
#         print(f"Epoch {epoch + 1}/{config.epochs}, "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#     print(f"Best model: {best_model_path}, Best val acc: {best_acc:.4f}")
#     with open(os.path.join(config.model_dir, "best_model.txt"), "w") as f:
#         f.write(best_model_path)
#
#
# if __name__ == '__main__':
#     config = CNN_config()
#     train_SwinTransformerV2(config)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm
from PIL import Image
import os
from dataclasses import dataclass


@dataclass
class CNN_config:
    model_name: str = "swinv2_tiny_window16_256"  # 모델 이름
    train_dir: str = "./train"  # 학습 데이터 경로
    test_dir: str = "./test"  # 테스트 데이터 경로
    model_dir: str = "./model"  # 모델 저장 경로
    mode: str = "train"  # "train" 또는 "test"
    batch_size: int = 32  # 배치 사이즈
    learning_rate: float = 0.001  # 학습률
    epochs: int = 100  # 학습 횟수
    pretrained: bool = False  # 사전 학습된 모델 사용 여부
    class_file: str = "./product.txt"  # 클래스 파일 경로
    image_size: (int, int) = (256, 256)  # 이미지 크기 (모델과 일치)
    normalize_mean: (float, float, float) = (0.485, 0.456, 0.406)  # 정규화 평균값
    normalize_std: (float, float, float) = (0.229, 0.224, 0.225)  # 정규화 표준편차
    num_workers: int = 0  # 데이터 로드에 사용할 CPU 코어 수
    weight_decay: float = 1e-4  # 가중치 감소 파라미터
    early_stopping_patience: int = 10  # 조기 종료 patience

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join("model", self.model_name)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.class_file, "r") as f:
            self.classes = f.read().splitlines()

        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}


def read_images_and_labels(config):
    if config.mode == "train":
        data_dir = config.train_dir
    elif config.mode == "test":
        data_dir = config.test_dir
    else:
        raise ValueError("Mode must be 'train' or 'test'.")

    print(f"Reading images and labels from {data_dir} ({config.mode})")

    images = []
    labels = []

    category = config.classes

    for c in category:
        path = os.path.join(data_dir, c)
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist.")
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(img_path)
                labels.append(c)

    print(f"Reading images and labels done. {len(images)} images found.")
    return images, labels


def create_model(config):
    model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=config.num_classes)
    return model


class ImageDataset(Dataset):
    def __init__(self, images, labels, config, transform=None):
        self.images = images
        self.labels = labels
        self.class_to_idx = config.class_to_idx
        self.transform = transform
        self.image_size = config.image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_name = self.labels[idx]
        label = self.class_to_idx[label_name]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, self.image_size[0], self.image_size[1])

        return image, label


def create_dataloader(config, images, labels):
    if config.mode == "train":
        data_transforms = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])

    dataset = ImageDataset(images, labels, config, transform=data_transforms)

    if config.mode == "train":
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers)

        return train_loader, val_loader
    else:
        test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        return test_loader


def train_SwinTransformerV2(config):
    images, labels = read_images_and_labels(config)
    train_loader, val_loader = create_dataloader(config, images, labels)

    model = create_model(config)
    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_acc = 0.0
    best_model_path = None
    epochs_no_improve = 0
    early_stop = False

    print(f"Start training {config.model_name}...")
    for epoch in range(config.epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels_batch in train_loader:
            inputs = inputs.to(config.device)
            labels_batch = labels_batch.to(config.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels_batch.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(config.device)
                labels_batch = labels_batch.to(config.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels_batch.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(config.model_dir, f"{config.model_name}_{epoch + 1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # if epochs_no_improve >= config.early_stopping_patience:
        #     print(
        #         f"Validation accuracy did not improve for {config.early_stopping_patience} epochs. Stopping training.")
        #     early_stop = True

        print(f"Epoch {epoch + 1}/{config.epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print(f"Best model: {best_model_path}, Best val acc: {best_acc:.4f}")
    with open(os.path.join(config.model_dir, "best_model.txt"), "w") as f:
        f.write(best_model_path)


if __name__ == '__main__':
    config = CNN_config()
    if config.mode == "train":
        train_SwinTransformerV2(config)

