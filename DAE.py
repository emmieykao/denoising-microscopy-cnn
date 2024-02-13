import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from save_data import DenoisingData
from random import randint
import cv2
from PIL import Image

class DAE(nn.Module):
    # reference: https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch
    def __init__(self, depth=5, n_channels=32, image_channels=1, 
        use_bnorm=False, kernel_size=3):
        super(DAE, self).__init__()
        kernel_size = 3
        padding = 1
        encoder_layers = []
        decoder_layers = []

        encoder_layers.append(nn.Conv2d(image_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=True))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Conv2d(n_channels, n_channels * 2, 
            kernel_size=kernel_size, padding=padding, bias=True))
        encoder_layers.append(nn.ReLU())
        # for _ in range(depth-2):
        #     encoder_layers.append(nn.Conv2d(n_channels * 4, 
        #     n_channels * 4, kernel_size=kernel_size, padding=padding, 
        #     bias=False))
        #     if use_bnorm:
        #         encoder_layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, 
        #         momentum = 0.95))
        #     encoder_layers.append(nn.ReLU(inplace=True))
        # encoder_layers.append(nn.Conv2d(n_channels * 4, 
        # n_channels * 8, kernel_size=kernel_size, padding=padding, 
        # bias=True))
        # encoder_layers.append(nn.ReLU(inplace=True))
        # encoder_layers.append(nn.Conv2d(n_channels * 8, 
        # n_channels * 16, kernel_size=kernel_size, padding=padding, 
        # bias=True))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers.append(nn.ConvTranspose2d(n_channels * 2, 
        n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.ConvTranspose2d(n_channels, 
        image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        decoder_layers.append(nn.ReLU(inplace=True))
        # decoder_layers.append(nn.ConvTranspose2d(n_channels * 4, 
        # n_channels * 2, kernel_size=kernel_size, padding=padding, bias=True))
        # decoder_layers.append(nn.ReLU(inplace=True))
        # decoder_layers.append(nn.ConvTranspose2d(n_channels * 2, 
        # n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        # decoder_layers.append(nn.ReLU(inplace=True))
        # decoder_layers.append(nn.ConvTranspose2d(n_channels * 4, 
        # image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        # decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

        # print(f'model summary: {self.encoder.summary}')

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def train_model(epochs, dataloader, model):
    loss_function = torch.nn.MSELoss()
    outputs = []
    losses = []
    for epoch in range(epochs):
        for (image, _) in dataloader:
        
            
            # Output of Autoencoder
            reconstructed = model(image)
            
            # Calculating the loss function
            loss = loss_function(reconstructed, image)
            print(f"{loss}")
            
            # Storing the losses in a list for plotting
            losses.append(loss)
        print(f"{epoch=}------------------------------")
    return model

def test_model(model):
    loss_function = torch.nn.MSELoss()
    test_dataset = DenoisingData("/Users/emmiekao/denoising-fluorescence/denoising/dataset", False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    losses = []
    save_img = True
    for i, test_object in enumerate(test_dataloader):
        test_image = test_object[0]
        test_clean = test_object[1]
        denoise_image = model.forward(test_image)
        print(denoise_image.shape)
        loss = loss_function(denoise_image, test_clean)
        denoise_image = denoise_image.cpu().detach().numpy()
        losses.append(loss)


        for j in range(denoise_image.shape[0]):
            if save_img:
            # cv2.imshow("denoised", denoise_image)
                im_output = Image.fromarray(denoise_image[j,0,:,:])
                im_output = im_output.convert("L")
                im_output.save(f"denoised_{i}_{j}.jpeg")
        save_img = False
            
            
        
        



def main():
    retrain = False
    model_path = "/Users/emmiekao/Desktop/denoising_project/torch_model.pt"
    # Model Initialization
    model = DAE()
    
    if os.path.exists(model_path) and not retrain:
        model = torch.load(model_path)
    else:
        train_dataset = DenoisingData("/Users/emmiekao/denoising-fluorescence/denoising/dataset", True)
        dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=1, drop_last=False)
        model = train_model(1, dataloader, model)
        torch.save(model, model_path)

    test_model(model)

if __name__ == '__main__':
    main()