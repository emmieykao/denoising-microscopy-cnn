import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/Users/emmiekao/Desktop/denoising_project/data/"
MODEL_EPOCHS = [20, 40, 60, 80, 100] # amount of epochs model was trained on
TOTAL_EPOCHS = 100


TRAINED = False # whether to load training or testing data

def import_data(filename: str) -> list:
    with open(filename, "rb") as f:
        new_data = pickle.load(f)
    return new_data

def plot_test(data_dict: dict, error: str) -> None:
    """
    creates plots for the testing PSNR, RMSE, or SSIM, graphing epochs trained on vs error

    Args:
    data_dict (dict): dictionary with data name as key
    error (str): "rmse" or "psnr" or "ssim"
    """
    filename = f"test_graph_{error}"
    raw = []
    for key in data_dict:
        if error in key:
            raw.append(float(data_dict[key][0]))

    print(raw)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(MODEL_EPOCHS, raw)
    ax.plot(MODEL_EPOCHS, raw)
    ax.set_xlim(xmin=0.0, xmax=110)
    if error == "ssim":
        ax.set_ylim(ymin=0, ymax=1)
    else:
        ax.set_ylim(ymin=0.0, ymax=90)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(error.upper())
    ax.set_title(f"Epochs vs. Testing {error.upper()}")
    plt.savefig(filename)


def plot_train(data_dict: dict, error: str, version: str) -> None:
    """
    creates plots for the training and validation PSNR and RMSE, graphing epochs vs error

    Args:
    data_dict (dict): dictionary with data name as key
    error (str): "rmse" or "psnr" or "ssim"
    version (str): "train" or "val" (training or validation data)
    """
    filename = f"{version}_graph_{error}"
    raw = []
    for key in data_dict:
        if "rmse" in key and version in key:
            raw = np.float_(data_dict[key])
    

    if error == "psnr":
        # calculate PSNR using RMSE
        raw = np.array(raw)
        raw = raw ** 2
        inner = 255**2 / raw
        inner = 10 * np.log10(inner)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(list(range(TOTAL_EPOCHS)), inner)
        ax.plot(list(range(TOTAL_EPOCHS)), inner)
        ax.set_xlim(xmin=0.0, xmax=110)
        ax.set_ylim(ymin=0.0, ymax=90)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(error.upper())
        if version == "train":
            ax.set_title(f"Epochs vs. Training {error.upper()}")
        else:
            ax.set_title(f"Epochs vs. Validation {error.upper()}")
        plt.savefig(filename)
        return
    else:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(list(range(TOTAL_EPOCHS)), raw)
        ax.plot(list(range(TOTAL_EPOCHS)), raw)
        ax.set_xlim(xmin=0.0, xmax=110)
        ax.set_ylim(ymin=0.0, ymax=90)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(error.upper())
        if version == "train":
            ax.set_title(f"Epochs vs. Training {error.upper()}")
        else:
            ax.set_title(f"Epochs vs. Validation {error.upper()}")
        plt.savefig(filename)
    

def main():
    data_dict = {}
    if TRAINED:
        file_names = [
            "train_psnr_list",
            "val_psnr_list",
            "train_rmse_list",
            "val_rmse_list",
        ]
        for name in file_names[:4]:
            path = f"{DATA_PATH}{name}.pkl"
            data_dict[name] = import_data(path)
        # choose whether to graph RMSE or PSNR and training ("train") or validation ("val") data 
        plot_train(data_dict, "rmse", "val")
    else:
        file_names = []
        for epoch in MODEL_EPOCHS:
            file_names.append(f"test_psnr_list_{epoch}")
            file_names.append(f"test_rmse_list_{epoch}")
            file_names.append(f"test_ssim_list_{epoch}")
        for name in file_names:
            path = f"{DATA_PATH}{name}.pkl"
            data_dict[name] = import_data(path)
        plot_test(data_dict, "ssim")


if __name__ == "__main__":
    main()



