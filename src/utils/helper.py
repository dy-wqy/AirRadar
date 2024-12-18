from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
import numpy as np
import os
from src.utils.scaler import StandardScaler


def get_pos(datapath):
    data = np.load(datapath + "/pos.npy")
    longitude_scaler = StandardScaler(mean=data[:, 0].mean(), std=data[:, 0].std())
    latitude_scaler = StandardScaler(mean=data[:, 1].mean(), std=data[:, 1].std())
    data[:, 0] = longitude_scaler.transform(data[:, 0])
    data[:, 1] = latitude_scaler.transform(data[:, 1])
    # standardized_longitude = longitude_scaler.transform(data[:, 0].reshape(-1, 1))

    # latitude_scaler.fit(data[:, 1].reshape(-1, 1))
    # standardized_latitude = latitude_scaler.transform(data[:, 1].reshape(-1, 1))
    # standardized_data = np.hstack((standardized_longitude, standardized_latitude))
    return data


def get_dataloader(datapath, batch_size, output_dim, mask_rate, mode="train"):
    """
    get data loader from preprocessed data
    """
    data = {}
    mask_nodes = {}
    processed = {}
    results = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(datapath, category + ".npz"))
        history_data = np.load(os.path.join(datapath, category + "_history.npy"))
        data["history_" + category] = history_data
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]

    for category in ["val", "test"]:
        cat_data = np.load(os.path.join(datapath, category + "_nodes.npy"))
        mask_node_num = len(cat_data)
        if category == "val":
            mask_nodes[category] = cat_data[0 : int(mask_node_num * mask_rate)]
        else:
            mask_nodes[category] = cat_data[0 : int(mask_node_num * mask_rate)]

    scalers = []
    # scalers.append(
    #     StandardScaler(
    #         mean=data["x_train"][..., output_dim].mean(),
    #         std=data["x_train"][..., output_dim].std(),
    #     )
    # )
    for i in range(output_dim):
        scalers.append(
            StandardScaler(
                mean=data["x_train"][..., i].mean(), std=data["x_train"][..., i].std()
            )
        )

    # Data format
    for category in ["train", "val", "test"]:
        # data["history_" + category][..., output_dim] = scalers[0].transform(
        #     data["history_" + category][..., output_dim]
        # )
        # data["x_" + category][..., output_dim] = scalers[0].transform(
        #     data["x_" + category][..., output_dim]
        # )
        # data["y_" + category][..., output_dim] = scalers[0].transform(
        #     data["y_" + category][..., output_dim]
        # )
        for i in range(output_dim):
            data["history_" + category][..., i] = scalers[i].transform(
                data["history_" + category][..., i]
            )
            data["x_" + category][..., i] = scalers[i].transform(
                data["x_" + category][..., i]
            )
            # data["y_" + category][..., i] = scalers[i].transform(
            #     data["y_" + category][..., i]

        new_x = Tensor(data["x_" + category])
        new_y = Tensor(data["y_" + category])
        new_history = Tensor(data["history_" + category])
        processed[category] = TensorDataset(new_x, new_y, new_history)

    results["train_loader"] = DataLoader(processed["train"], batch_size, shuffle=True)
    results["val_loader"] = DataLoader(processed["val"], batch_size, shuffle=False)
    results["test_loader"] = DataLoader(processed["test"], batch_size, shuffle=False)

    print(
        "train: {}\t valid: {}\t test:{}".format(
            len(results["train_loader"].dataset),
            len(results["val_loader"].dataset),
            len(results["test_loader"].dataset),
        )
    )
    results["scalers"] = scalers
    return (results, mask_nodes)


def check_device(device=None):
    if device is None:
        print(
            "`device` is missing, try to train and evaluate the model on default device."
        )
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        print(torch.device)
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)


def get_num_nodes(dataset):
    d = {"AIR_TINY": 1085}
    assert dataset in d.keys()
    return d[dataset]
