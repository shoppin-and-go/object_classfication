import os


def count_imgs(path):
    count = 0
    for f in os.listdir(path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            count += 1
    return count


def count_train_imgs():
    label_txt = "./product.txt"
    train_dir = "./train"

    labels = []

    with open(label_txt, "r") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)

    for label in labels:
        path = os.path.join(train_dir, label)
        print(f"{label}: {count_imgs(path)} images")


if __name__ == '__main__':
    count_train_imgs()
