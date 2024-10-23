import os


def make_folder(file="./product.txt", target_dir="./train"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(file, "r") as f:
        for line in f:
            folder_name = line.strip()
            if folder_name:
                folder_path = os.path.join(target_dir, folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)


if __name__ == '__main__':
    make_folder()
