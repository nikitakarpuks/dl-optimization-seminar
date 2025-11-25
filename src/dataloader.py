from torch.utils.data import DataLoader


def get_dataloader(dataset, config):
    batch_size = config["batch_size"]
    trainloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, num_workers=0)

    dataloader = {'train': trainloader, 'test': testloader}

    return dataloader
