import torch
from torch import nn
import yaml
from src.dataset import get_dataset
from src.dataloader import get_dataloader
from src.visualizations import visualize_dataset
from src.dense_network import MyNetwork


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    cfg = yaml.load(open('./config/config.yml', 'r'), Loader=yaml.FullLoader)

    dataset = get_dataset(cfg["dataset"])

    visualize_dataset(cfg, dataset["test"])

    dataloader = get_dataloader(dataset, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"])

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(dataloader["train"], model, loss_fn, optimizer)
        test_loop(dataloader["eval"], model, loss_fn)
    test_loop(dataloader["test"], model, loss_fn)
    print("Done!")

    #
    # criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    # for t in range(30000):
    #     # Forward pass: Compute predicted y by passing x to the model
    #     y_pred = model(x)
    #
    #     # Compute and print loss
    #     loss = criterion(y_pred, y)
    #     if t % 2000 == 1999:
    #         print(t, loss.item())
    #
    #     # Zero gradients, perform a backward pass, and update the weights.
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # print(f'Result: {model.string()}')


if __name__ == '__main__':
    main()
