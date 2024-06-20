Image-to-Image Translation using Pix2Pix Models
===

This project implements an image-to-image translation model using the Pix2Pix architecture [1] with TensorFlow. The goal is to generate realistic images from input images using the TU-Graz dataset. The performance of the model is evaluated using several quantitative metrics (PSNR, SSIM, Cosine Similarity, FCN-Score) as well as qualitative analysis.

## Dataset Description

The [Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387) (TU-Graz) dataset [2] is used for training and evaluating the model. This dataset contains we used the Semantic Drone Dataset focused on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts more than 20 houses from nadir (bird’s eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 400x600.

## Results comparison

![Results comparison](grid_image.png)

## Install the environment in a local device
The following steps must be followed to install the dependencies required for running the application:

1. Navigate to the project directory
```
cd (`project_path`)
```

2. Create a conda environment from a .yml file
```
conda env create -f environment.yml
```

## Project Structure
The project is organized as follows:

```
├── datasets
├── i-t-i_translation.ipynb
├── train_segmentation.ipynb
```

## Dependencies
The main libraries used in this project include:

- Tensorflow
- NumPy
- Matplotlib
- Keras
- scikit-image

## Methodology
### Model Architecture
The Pix2Pix model consists of a generator and a discriminator:

- Generator: Transforms input images to output images.
- Discriminator: Distinguishes between real and generated images.

### Training Process
The training process involves:

- Loading the dataset: Using the data loader to load and preprocess images.
- Training the generator and discriminator: Using adversarial loss and reconstruction loss to train the model.
- Saving the model: Periodically saving the model checkpoints for evaluation.

### Evaluation Metrics
- PSNR
Peak Signal-to-Noise Ratio (PSNR) measures the quality of the generated images compared to the ground truth. Higher PSNR indicates better image quality.

- SSIM
Structural Similarity Index (SSIM) measures the similarity between the generated images and the ground truth. SSIM considers changes in structural information, luminance, and contrast.

- Cosine Similarity
Cosine Similarity measures the similarity between two images by calculating the cosine of the angle between their feature vectors. Higher cosine similarity indicates more similar images.

- FCN-Score
FCN-Score evaluates the segmentation performance of the generated images using a Fully Convolutional Network (FCN). Higher FCN-Score indicates better segmentation quality.

- Qualitative Analysis
Qualitative analysis involves visual inspection of the generated images to assess their realism. This can include side-by-side comparisons with ground truth images and evaluation by human observers.

## References

[1] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. CVPR, 2017.

[2] Safe-UAV-Landing. https://www.tugraz.at/index.php?id=22387, 2021. [Accessed 20-March-2024].
