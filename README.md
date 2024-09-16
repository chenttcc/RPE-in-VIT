# Vision Transformer (ViT) with RPE Project

## Project Overview
This project focuses on improving the functionality of the Vision Transformer (ViT) by integrating relative position encoding (RPE) and experimenting with large pre-trained models to enhance performance. The project explores two transformer architectures: a regular (vanilla) transformer and an improved transformer known as Performer. The goal is to enhance image classification performance on two commonly used datasets: MNIST and TinyImageNet.

### Target
The primary aims of this project are to:
- Implement and compare the performance of regular and Performer transformer architectures for image classification tasks.
- Integrate relative position encoding in both architectures using a mask matrix.
- Explore the efficacy of using large pre-trained models to significantly improve classification accuracy.

## Relative Position Encoding (RPE)
Relative position encoding is a method used to incorporate location information in the image during the transformer’s processing. This technique utilizes a two-dimensional position encoding matrix, where each element represents the distance between two pixels in the image. RPE enhances the model's ability to account for spatial relationships between pixels, which improves the accuracy of image classification.

In this project, three different methods for constructing the relative position encoding matrix are implemented:

1. **General RPE**: 
   - Each element in the matrix is a learnable parameter of the neural network. No specific function is used to calculate the matrix; the model automatically learns the values during training.

2. **Exponential Function RPE**:
   - This method assumes that the larger the distance between two pixels, the smaller the impact they have on each other. The distance between pixels is transformed using an exponential function to model this relationship.

3. **Polynomial Function RPE**:
   - In this approach, the distance between pixels is transformed using \( \frac{1}{\text{polynomial}} \) to model the decreasing impact. This method provides another way to incorporate distance-related effects between pixels.

All three RPE methods are implemented in both the regular and Performer transformer architectures, and experimental results demonstrate that applying RPE improves accuracy compared to transformers without RPE. However, the general accuracy achieved using these methods is capped at around 32%.

## Large Pre-Trained Model Fine-Tuning
To further enhance performance, this project also incorporates a large pre-trained model from [Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k). Fine-tuning this pre-trained model on the TinyImageNet datasets significantly boosts accuracy, with the highest recorded accuracy reaching 88%. 

A notebook is provided within the project to demonstrate the fine-tuning process, including the steps for training and validating the model. The results showcase that using a pre-trained model can drastically improve accuracy in image classification tasks compared to using ViTs from scratch.

## Project Structure
The project is organized into the following folders:
1. **Regular Transformer**: 
   - Contains the source code for implementing a standard vanilla transformer architecture for image recognition with quadratic complexity.
   - The RPE technique is included but can be toggled on or off by the user.

2. **Performer**:
   - Contains the source code for the Performer architecture, which reduces the quadratic complexity to subquardratic.
   - The RPE technique is implemented, and users can decide whether to apply it during training.

3. **Pre-Trained Model**:
   - Contains a notebook that demonstrates the fine-tuning process of the large pre-trained model. This section includes training results and performance metrics showing the improved accuracy achieved through fine-tuning.

4. **Datasets**:
   - Provides usage and test cases for the MNIST and TinyImageNet datasets.

## Main Features
- **Relative Position Encoding**: The project incorporates three RPE methods (General RPE, Exponential RPE, and Polynomial RPE) into both regular and Performer transformer architectures, enhancing image classification accuracy.
- **Quadratic Complexity Optimization**: While the regular transformer operates with quadratic complexity, the Performer architecture is implemented to achieve subquadratic complexity, thus improving computational efficiency.
- **Pre-Trained Model Fine-Tuning**: The project demonstrates how a large pre-trained model can be fine-tuned on the MNIST and TinyImageNet datasets to achieve an accuracy of up to 88%, showcasing significant improvement over standard transformers.

## Datasets
- **MNIST**: A widely used dataset for handwritten digit classification.
- **TinyImageNet**: A more complex dataset derived from the larger ImageNet dataset, used for benchmarking image classification tasks. [Hugging Face](https://huggingface.co/datasets/zh-plus/tiny-imagenet)


## References
- The **Performer** paper and source code: https://arxiv.org/pdf/2009.14794
- The **RPE** technique in paper: https://arxiv.org/pdf/2107.07999
- The **VIT** paper: https://arxiv.org/pdf/2010.11929


