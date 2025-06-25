# GAN-Based Masked Face Unmasking Network

## Description

This repository contains the implementation of a Generative Adversarial Network (GAN)-based approach for the "unmasking" of masked faces, inspired by recent advancements in image completion and generative models. The primary goal is to effectively remove facial masks from images and plausibly synthesize the occluded facial regions, maintaining global coherence and fine details.

This project specifically focuses on creating synthetic training data and implementing a U-Net-like Generator and PatchGAN Discriminators to achieve high-quality face completion in masked areas.

## Dataset Information

The model is trained using a composite dataset:

1.  **CelebA Dataset:** The base dataset for original, unmasked face images. This provides a rich variety of human faces.
2.  **Synthetically Generated Masked Dataset:** Due to the lack of paired masked/unmasked face images, a synthetic dataset is generated on the fly.
    * **Process:** Faces from the CelebA dataset are first detected using `dlib`'s frontal face detector and facial landmark predictor, then aligned and cropped to a uniform size (256x256).
    * **Masking:** Simple geometric masks (rectangular or oval) are randomly applied to the aligned faces to simulate facial masks.
    * **Output:** For each original image, a corresponding masked image and a binary mask map (indicating the masked region) are generated and saved.
    * **Data Directories:**
        * `data/original_celeba_aligned/`: Aligned and cropped original CelebA faces.
        * `data/masked_images/`: Faces with synthetic masks applied.
        * `data/mask_maps/`: Binary mask maps corresponding to the masked regions.
    * **Number of Synthetic Data:** Configurable, default is `10000` samples.

## Model Architecture

The core of the unmasking network comprises a Generator and two Discriminators, following the principles of GANs for image-to-image translation tasks.

### 1. Generator (`G_edit`)

* **Architecture:** A U-Net like encoder-decoder structure with skip connections.
* **Input:** Concatenated tensor of the masked face image (3 channels) and its corresponding binary mask map (1 channel), totaling 4 input channels.
* **Output:** A 3-channel RGB image representing the unmasked (completed) face.
* **Layers:** Utilizes `Conv2d` for downsampling, `ConvTranspose2d` for upsampling, `BatchNorm2d`, `LeakyReLU` in encoder, and `ReLU` in decoder. Skip connections help preserve fine details from the encoder path.

### 2. Discriminators (`netD_whole`, `netD_mask`)

Two separate discriminators are employed to provide robust adversarial training:

* **`netD_whole` (Whole Image Discriminator):**
    * **Architecture:** A standard Convolutional Neural Network (CNN) based PatchGAN discriminator.
    * **Input:** A 3-channel RGB image (either real unmasked image or generated unmasked image).
    * **Purpose:** Discriminates between real (original) faces and generated (unmasked) faces across the entire image.
* **`netD_mask` (Mask Region Discriminator):**
    * **Architecture:** Similar CNN-based PatchGAN discriminator.
    * **Input:** A 3-channel RGB image where the unmasked region is from the input, and the masked region is either from the original (ground truth) or the generated image.
    * **Purpose:** Focuses specifically on the plausibility of the synthesized mask region.

## Hyperparameters

The training process is configured with the following key hyperparameters:

* **Image Size (`IMG_SIZE`):** 256x256 pixels
* **Batch Size (`BATCH_SIZE`):** 16
* **Number of Epochs (`NUM_EPOCHS`):** 100
* **Learning Rate (Generator `LEARNING_RATE_G`):** 0.0001
* **Learning Rate (Discriminators `LEARNING_RATE_D`):** 0.0001
* **Adam Optimizer Betas (`BETA1`, `BETA2`):** (0.5, 0.999)

### Loss Function Weights:

The total Generator loss (`errG`) is a weighted sum of multiple components:

* **Reconstruction Loss Weight (`LAMBDA_RC`):** 100.0 (Combines L1 Loss and SSIM Loss)
* **Perceptual Loss Weight (`LAMBDA_PERC`):** 1.0
* **Adversarial Loss Weights:**
    * Whole Region (`LAMBDA_ADV_WHOLE_REGION`): 1.0
    * Mask Region (`LAMBDA_ADV_MASK_REGION`): 1.0

## Training and Validation Process

### Training Setup:

* **Device:** Automatically detects and uses GPU (`cuda`) if available, otherwise CPU.
* **Optimizers:** Adam optimizer is used for both the Generator and the two Discriminators.
* **Loss Functions:**
    * **Reconstruction Loss:** Combination of Mean Absolute Error (L1 Loss) and Structural Similarity Index Measure (SSIM) between generated and ground truth images.
    * **Perceptual Loss:** Utilizes features extracted from a pre-trained VGG16 network (`models.vgg16`) to measure perceptual similarity between images.
    * **Adversarial Loss:** Binary Cross-Entropy with Logits Loss (`nn.BCEWithLogitsLoss`) for the GAN training.
* **Weights Initialization:** All network weights are initialized from a normal distribution.
* **Checkpointing:** The model and optimizer states are saved periodically (`SAVE_INTERVAL` epochs) to allow for training resumption. The training process can automatically load the latest checkpoint to resume from a specific epoch.

### Training Loop:

The training iterates through a specified number of epochs and batches:

1.  **Discriminator Update:**
    * `netD_whole` is trained to distinguish between real unmasked images (`I_gt`) and generated unmasked images (`I_edit`).
    * `netD_mask` is trained to distinguish between the real mask region (from `I_gt`) and the generated mask region (from `I_edit`).
2.  **Generator Update:**
    * `netG` is trained to generate realistic unmasked images that can fool both `netD_whole` and `netD_mask`.
    * The Generator's loss is a weighted sum of:
        * Reconstruction Loss (`loss_rc`)
        * Perceptual Loss (`loss_perc`)
        * Adversarial Loss from `netD_whole` (`loss_adv_whole_region`)
        * Adversarial Loss from `netD_mask` (`loss_adv_mask_region`)

### Evaluation:

After training, the Generator's performance is evaluated using standard image quality metrics:

* **PSNR (Peak Signal-to-Noise Ratio):** Measures the quality of reconstruction.
* **SSIM (Structural Similarity Index Measure):** Evaluates the perceived quality and structural similarity between images.
* **Perceptual Loss (FD - Feature Distance):** Measures the difference in high-level features, indicating perceptual realism.

Sample unmasking results are visualized, comparing original, masked, and edited images, along with their PSNR, SSIM, and Perceptual Loss scores.

---

### **Evaluation Results**

**Note:** The following results are based on the evaluation of 5 randomly selected images from the dataset after training up to epoch 95.

| Metric                 | Average Value           |
| :--------------------- | :---------------------- |
| PSNR                   | 35.64 dB                |
| SSIM                   | 0.9611                  |
| Perceptual Loss (FD)   | 0.4351                  |

**Individual Sample Results & Visual Examples:**

| Original Image | Masked Image | Edited Image (Unmasked) |
| :------------- | :----------- | :---------------------- |
| ![Original 1](path/to/original_1.jpg) | ![Masked 1](path/to/masked_1.jpg) | ![Edited 1](path/to/edited_1.jpg) |
| _PSNR: 35.52, SSIM: 0.9667, PercLoss: 0.4126_ | | |
| ![Original 2](path/to/original_2.jpg) | ![Masked 2](path/to/masked_2.jpg) | ![Edited 2](path/to/edited_2.jpg) |
| _PSNR: 37.68, SSIM: 0.9687, PercLoss: 0.4400_ | | |
| ![Original 3](path/to/original_3.jpg) | ![Masked 3](path/to/masked_3.jpg) | ![Edited 3](path/to/edited_3.jpg) |
| _PSNR: 32.46, SSIM: 0.9290, PercLoss: 0.5458_ | | |
| ![Original 4](path/to/original_4.jpg) | ![Masked 4](path/to/masked_4.jpg) | ![Edited 4](path/to/edited_4.jpg) |
| _PSNR: 36.57, SSIM: 0.9750, PercLoss: 0.3728_ | | |
| ![Original 5](path/to/original_5.jpg) | ![Masked 5](path/to/masked_5.jpg) | ![Edited 5](path/to/edited_5.jpg) |
| _PSNR: 35.95, SSIM: 0.9660, PercLoss: 0.4041_ | | |
*(**Important:** Please replace `path/to/original_1.jpg` etc. with the actual URLs of your images on your GitHub repository after you upload them.)*

---

## Improvements and Future Updates

* **More Diverse Mask Generation:** Explore semantic mask generation (e.g., using a mask R-CNN) or more complex, realistic mask shapes and textures.
* **Real-world Masked Face Evaluation:** Test the model's performance on real-world masked face datasets, which often present greater variability in lighting, pose, and mask types.
* **Alternative GAN Architectures:** Experiment with advanced GAN architectures like StyleGAN2 or SPADE for potentially higher-fidelity image synthesis.
* **Quantifying Mask Detection:** Integrate a robust mask detection module and evaluate its performance separately.
* **Deployment and Inference:** Develop a user-friendly interface or API for easy inference on new masked face images.

## Conclusion

This project demonstrates a robust GAN-based framework for masked face unmasking, capable of synthesizing plausible facial features in occluded regions. By combining reconstruction, perceptual, and adversarial losses, the model learns to generate high-quality, perceptually realistic unmasked faces.

## Acknowledgments

This project's approach and methodology are inspired by the research presented in the paper:

* **"A Novel GAN-Based Network for Unmasking of Masked Face"**
    * **Authors:** Nizam Ud Din, Kamran Javed, Seho Bae, and Juneho Yi
    * **Publication:** IEEE Access
    * This work was conducted at Sungkyunkwan University, Suwon, South Korea.
