import matplotlib.pyplot as plt
import torch


def visualize_dataset(cfg, data):

    labels_map = {v: k for k, v in data.class_to_idx.items()}

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        if cfg["dataset"] == "mnist":
            plt.imshow(img, cmap="gray")
        elif cfg["dataset"] == "cifar10":
            plt.imshow(img.permute(1, 2, 0))
        else:
            raise NotImplementedError
    plt.show()
    plt.close()
