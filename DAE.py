import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.nn.functional as F
from save_data import DenoisingData
from random import randint
import cv2
from PIL import Image
from tqdm import tqdm
from metrics import *
import math
import pickle
from sklearn.model_selection import KFold
import random

DATA_PATH = "/Users/emmiekao/denoising-fluorescence/denoising/dataset"
MODEL_EPOCHS = 100 # amount of epochs model was trained on
MODEL_PATH = f"/Users/emmiekao/Desktop/denoising_project/torch_model_{MODEL_EPOCHS}.pt"

DATA_SAVE_PATH = "/Users/emmiekao/Desktop/denoising_project/data/"
LOSS_FUNC = "ssim" # choose from ["psnr", "ssim"]
RETRAIN = False
IMG_PIXELS = 2**18
random.seed(1)
np.random.seed(1)

class DAE(nn.Module):
    def __init__(self, layers=3, n_channels=32, image_channels=1, kernel_size=3,
    padding=1):
        super(DAE, self).__init__()
        encoder_layers = []
        decoder_layers = []

        for l in range(layers):
            if l == 0:
                encoder_layers.append(nn.Conv2d(image_channels, n_channels, 
                    kernel_size=kernel_size, padding=padding, bias=True))
                encoder_layers.append(nn.ReLU())
            else:
                encoder_layers.append(nn.Conv2d(n_channels * 2**(l-1), n_channels * 2**l, 
                    kernel_size=kernel_size, padding=padding, bias=True))
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        for l in range(layers):
            if l == layers - 1:
                decoder_layers.append(nn.ConvTranspose2d(n_channels, 
                    image_channels, kernel_size=kernel_size, padding=padding, bias=True))
                decoder_layers.append(nn.ReLU(inplace=True))
            else:
                decoder_layers.append(nn.ConvTranspose2d(n_channels * 2**(layers-l-1), 
                    n_channels * 2**(layers-l-2), kernel_size=kernel_size, padding=padding, bias=True))
                decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def train_model(epochs, dataloader, model, device):
    psnr_list = []
    rmse_list = []
    val_psnr_list = []
    val_rmse_list = []
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001) #, weight_decay=1e-5) --> could also construct a specific class for this
    valid_dataset = DenoisingData(DATA_PATH, 2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    
    
    outputs = []
    losses = []
    for epoch in range(epochs):
        psnr, mse = 0., 0.
        for train_object in dataloader:
            train_image_list = train_object[0]
            train_clean = train_object[1]
            for j in range(train_image_list.shape[0]):
                indices=torch.LongTensor([j])
                train_image = train_image_list.index_select(0, indices)
                reconstructed = model(train_image)
                # Calculating the loss function
                loss = F.mse_loss(reconstructed, train_clean.index_select(0, indices), reduction="sum")
                # print(f"TRAINING mean squared error: {loss}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() # figure out why this is here :(
                # Storing the losses in a list for plotting
                mse += loss.item()
                if LOSS_FUNC == "psnr":
                    with torch.no_grad():
                        new_psnr = cal_psnr(train_clean.index_select(0, indices), reconstructed).sum().item()
                        # print(f"TRAINING psnr: {new_psnr}")
                        psnr += new_psnr
        
        # validation
        val_psnr, val_mse = 0., 0.
        for valid_object in valid_dataloader:
            valid_image_list = valid_object[0]
            valid_clean = valid_object[1]
            for j in range(valid_image_list.shape[0]):
                indices=torch.LongTensor([j])
                valid_image = valid_image_list.index_select(0, indices)
                reconstructed = model(valid_image)
                # Calculating the loss function
                loss = F.mse_loss(reconstructed, valid_clean.index_select(0, indices), reduction="sum")
                # print(f"VALIDATION mean squared error: {loss}")
                # Storing the losses in a list for plotting
                val_mse += loss.item()
                if LOSS_FUNC == "psnr":
                    with torch.no_grad():
                        new_val_psnr = cal_psnr(valid_clean.index_select(0, indices), reconstructed).sum().item()
                        # print(f"VALIDATION psnr: {new_psnr}")
                        val_psnr += new_val_psnr

        psnr = psnr / len(train_image_list)
        psnr_list.append(str(psnr))
        rmse = math.sqrt(mse / IMG_PIXELS)
        rmse_list.append(str(rmse))
        val_psnr = val_psnr / len(valid_image_list)
        val_psnr_list.append(val_psnr)
        val_rmse = math.sqrt(val_mse / IMG_PIXELS)
        val_rmse_list.append(str(val_rmse))
        
        
        print(f"{epoch=}------------------------------{time.strftime('%H:%M:%S')}")
    return model, psnr_list, val_psnr_list, rmse_list, val_rmse_list

def test_model(model):
    test_dataset = DenoisingData(DATA_PATH, 1)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    losses = []
    save_img = True
    psnr, mse = 0., 0.
    psnr_list = []
    rmse_list = []
    ssim = 0.
    ssim_list = []
    for i, test_object in enumerate(test_dataloader):
        test_image_list = test_object[0]
        test_clean_list = test_object[1]
        for j in range(test_image_list.shape[0]):
            indices=torch.LongTensor([j])
            test_image = test_image_list.index_select(0, indices)
            test_clean = test_clean_list.index_select(0, indices)
            reconstructed = model.forward(test_image)
        
            loss = F.mse_loss(reconstructed, test_clean, reduction = "sum")
            mse += loss.item()

            if LOSS_FUNC == "psnr":
                    with torch.no_grad():
                        new_psnr = cal_psnr(test_clean, reconstructed).sum().item()
                        psnr += new_psnr
            if LOSS_FUNC == "ssim":
                with torch.no_grad():
                        new_ssim = cal_ssim(test_clean, reconstructed).sum()
                        ssim += new_ssim
            reconstructed = reconstructed.detach().numpy()

            if save_img:
                # save noisy
                img_output = Image.fromarray(test_image[0,0,:,:].numpy())
                img_output = img_output.convert("L")
                img_output.save(f"images_{MODEL_EPOCHS}/noisy_{MODEL_EPOCHS}_{i}_{j}.jpeg")

                # save denoised
                dae_output = Image.fromarray(reconstructed[0,0,:,:])
                dae_output = dae_output.convert("L")
                dae_output.save(f"images_{MODEL_EPOCHS}/denoised_{MODEL_EPOCHS}_{i}_{j}.jpeg")

                # save clean
                clean_output = Image.fromarray(test_clean[0,0,:,:].numpy())
                clean_output = clean_output.convert("L")
                clean_output.save(f"images_{MODEL_EPOCHS}/clean_gt_{MODEL_EPOCHS}_{i}_{j}.jpeg")
                
        rmse = math.sqrt(mse / IMG_PIXELS)
        rmse_list.append(str(rmse))

        if LOSS_FUNC == "psnr":
            psnr = psnr / len(test_image_list)
            psnr_list.append(str(psnr))
            return psnr_list, rmse_list
        elif LOSS_FUNC == "ssim":
            ssim = ssim / len(test_image_list)
            ssim_list.append(str(ssim))
            return ssim_list, rmse_list

def train_simple(model, X_train, y_train, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    for epoch in range(epochs):
        for i, train_image in enumerate(X_train):
                reconstructed = model(train_image)
                # Calculating the loss function
                loss = F.mse_loss(reconstructed, y_train[i], reduction="sum")
                # print(f"TRAINING mean squared error: {loss}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    return model
        
def k_fold(epochs):
    X_train = []
    y_train = []

    # load data
    train_dataset = DenoisingData(DATA_PATH, 3)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=False)
    for train_object in dataloader:
        train_image_list = train_object[0]
        train_clean = train_object[1]
        for j in range(train_image_list.shape[0]):
            indices=torch.LongTensor([j])
            X_train.append(train_image_list.index_select(0, indices))
            y_train.append(train_clean.index_select(0, indices))

    # Set up K-Fold Cross-Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    scores = []
    for i in range(3):
        print(f'NUMER OF LAYERS: {i}---------------------------------')
        for train_index, val_index in kf.split(X_train):
            print(f'Training fold {fold_no}...')
            print(f'{train_index=}')
            X_train_fold = []
            y_train_fold = []
            X_val_fold = []
            y_val_fold = []
            for index in train_index:
                X_train_fold.append(X_train[index])
                y_train_fold.append(y_train[index])
            
            for index in val_index:
                X_val_fold.append(X_train[index])
                y_train_fold.append(y_train[index])

            model = DAE(layers=i) # fix this
            # Create and train a new instance of the CNN model
            model = train_simple(model, X_train_fold, y_train_fold, epochs)    
            # Evaluate the model on the validation fold
            score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            scores.append(score)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {score[0]}; {model.metrics_names[1]} of {score[1]*100}%')
            
            fold_no += 1

    # Calculate average scores across all folds
    average_score = np.mean(scores, axis=0)
    print(f'Average cross-validation score: {average_score[0]}')
    print(f'Average cross-validation accuracy: {average_score[1]*100}%')
        
            


def main():
    data = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    k_fold(1)

    # model = DAE()
    
    # if os.path.exists(MODEL_PATH) and not RETRAIN:
    #     model = torch.load(MODEL_PATH)
    # else:
    #     train_dataset = DenoisingData(DATA_PATH, 0)
    #     dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=False)
    #     model, train_psnr_list, val_psnr_list, train_rmse_list, val_rmse_list = train_model(20, dataloader, model, device)
    #     data["train_psnr_list"] = train_psnr_list
    #     data["val_psnr_list"] = val_psnr_list
    #     data["train_rmse_list"] = train_rmse_list
    #     data["val_rmse_list"] = val_rmse_list
    #     torch.save(model, MODEL_PATH)
    
    # if LOSS_FUNC == "psnr":
    #     test_psnr_list, test_rmse_list = test_model(model)
    #     data[f"test_psnr_list_{MODEL_EPOCHS}"] = test_psnr_list
    #     data[f"test_rmse_list_{MODEL_EPOCHS}"] = test_rmse_list
    # elif LOSS_FUNC == "ssim":
    #     test_ssim_list, test_rmse_list = test_model(model)
    #     data[f"test_ssim_list_{MODEL_EPOCHS}"] = test_ssim_list
    #     data[f"test_rmse_list_{MODEL_EPOCHS}"] = test_rmse_list

    # for dataset in data:
    #     with open(f"{DATA_SAVE_PATH}{dataset}.pkl", "wb") as f:
    #         pickle.dump(data[dataset], f)

if __name__ == '__main__':
    main()
