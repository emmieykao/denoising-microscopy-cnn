# Science Fair Project: Utilizing An Autoencoder To Denoise Confocal Microscopic Images

## Project Abstract:

Many areas of science and biology depend on microscopic images to determine molecular properties and the existence of microorganisms invisible to the human eye. By the nature of a microscopic lens, pixel sizes, and other factors, unwanted noise often finds its way into the microscopic images. Fluorescence microscopy, in particular, suffers from excess noise due to the bleaching of fluorophores. Thus, denoising such images is of great interest. In the past decade, artificial intelligence (AI) denoising algorithms experienced great success in image denoising; one of the most popular algorithms is the convolutional neural network (CNN), due to its ability to robustly process spatial information present in images. However, CNN autoencoders have been untested in microscopy imaging denoising tasks; further, many microscopy denoising algorithms are validated on unrealistic noise mechanisms (strictly Gaussian). Here, we use a CNN autoencoder to denoise confocal microscopic images with a realistic noise mechanism and apply the encoder layers to extract specific features in different types of cell images. 

We begin by cleaning and reorganizing a dataset with images containing Poisson-Gaussian noise, mimicking error that might be encountered in a real-life noisy image. Our final dataset contains (noisy, clean) pairs that the model can train on. We then construct a CNN autoencoder capable of denoising our microscopy images, yielding competitive PSNR and SSIM with other benchmarks. Next, we use the encoder half of the model to condense significant features of our images into two dimensions (and analyze biologically meaningful patterns from these encodings). Finally, we reflect on future uses and potential challenges for further research.

## File Description:
`metrics.py`: initializes PSNR and SSIM error functions used to test the accuracy of our model (credits to [https://github.com/yinhaoz/denoising-fluorescence](https://github.com/yinhaoz/denoising-fluorescence))
`save_data.py`: creates a PyTorch `dataset` class specific for our denoising data. We used data from [this repository](https://github.com/yinhaoz/denoising-fluorescence).<br>
`DAE.py`: constructs the encoder and decoder portions of our CNN autoencoder model. trains, tests, and validates the model using imagesets from before. saves error data into `.pkl` files for later use.
`graph_error.py`: uses `.pkl` files from `DAE.py` to graph error for our model.

`graphs`: contains error graphs from `graph_error.py`.
`models`: contains 20, 40, 60, 80, and 100-epoch models.

## Credits:
Used data from [https://github.com/yinhaoz/denoising-fluorescence](https://github.com/yinhaoz/denoising-fluorescence)
