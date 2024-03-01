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
    creates plots for the testing PSNR and RMSE, graphing epochs trained on vs error

    Args:
    data_dict (dict): dictionary with data name as key
    error (str): "rmse" or "psnr"
    """
    raw = []
    for key in data_dict:
        if error in key:
            raw.append(float(data_dict[key][0]))
    
    plt.scatter(MODEL_EPOCHS, raw)
    plt.plot(MODEL_EPOCHS, raw)
    plt.show()

def plot_train(data_dict: dict, error: str, version: str) -> None:
    """
    creates plots for the training and validation PSNR and RMSE, graphing epochs vs error

    Args:
    data_dict (dict): dictionary with data name as key
    error (str): "rmse" or "psnr"
    """
    raw = []
    for key in data_dict:
        if "rmse" in key and version in key:
            raw = np.float_(data_dict[key])
    

    if error == "psnr":
        raw = np.array(raw)
        raw = raw ** 2
        inner = 255**2 / raw
        inner = 10 * np.log10(inner)
        plt.scatter(list(range(TOTAL_EPOCHS)), inner)
        plt.plot(list(range(TOTAL_EPOCHS)), inner)
        plt.show()
        return
    else:
        plt.scatter(list(range(TOTAL_EPOCHS)), raw)
        plt.plot(list(range(TOTAL_EPOCHS)), raw)
        plt.show()
    

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
        plot_train(data_dict, "psnr", "val")
    else:
        file_names = []
        for epoch in MODEL_EPOCHS:
            file_names.append(f"test_psnr_list_{epoch}")
            file_names.append(f"test_rmse_list_{epoch}")
        for name in file_names:
            path = f"{DATA_PATH}{name}.pkl"
            data_dict[name] = import_data(path)
        plot_test(data_dict, "psnr")


if __name__ == "__main__":
    main()


# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(x_data,y_data)
# ax.set_xlim(xmin=0.0, xmax=1000)
# plt.savefig(filename)
