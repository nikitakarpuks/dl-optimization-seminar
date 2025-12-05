from torch.utils.data import DataLoader


def get_dataloader(dataset, config):
    batch_size = config["batch_size"]
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset["eval"], batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, num_workers=0)

    dataloader = {'train': train_loader, 'test': test_loader, 'eval': eval_loader}

    return dataloader
