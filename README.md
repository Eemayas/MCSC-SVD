# Embedding Watermark Project

## Project Title and Description

This project demonstrates the technique of embedding a watermark into a high-resolution image using Singular Value Decomposition (SVD). The watermark embedding process enhances image security by allowing the original image to be traced back through the extraction process. The project uses Python along with libraries like PIL, NumPy, and OpenCV to handle image processing tasks.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contact Information](#contact-information)

## Installation

To run this project, ensure you have Python installed along with the necessary libraries. Follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/Eemayas/MCSC-SVD.git
   cd MCSC-SVD
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the host image and watermark image saved in the local directory:
   - `host_image.jpg` for the host image
   - `watermark_image.jpg` for the watermark image

## Usage

Follow the steps below to use the project:

1. **Embedding the Watermark:**

   - Run the script to embed the watermark into the host image. The output will be saved as `output_image.png`.
   - The code performs SVD on both the host and watermark images, modifies the singular values of the host image, and then reconstructs the watermarked image.

2. **Extracting the Watermark:**

   - The script also demonstrates how to extract the watermark from the watermarked image.
   - The extracted watermark will be saved as `watermark_output_image.png`.

3. **Display Images:**
   - The images are displayed during the process using the `show()` method, allowing you to visualize the results.

## Features

- **Image Processing**: Utilizes the PIL and OpenCV libraries for image manipulation.
- **Singular Value Decomposition (SVD)**: Embeds and extracts watermarks using SVD, maintaining image integrity.
- **Customizable Alpha Parameter**: Adjust the `alpha` parameter to control the intensity of the watermark.
- **Grayscale Conversion**: Converts images to grayscale for simplicity and reduced computational load.

## Contact Information

For any questions or further information, feel free to reach out:

- **Name**: Prashant Manandhar
- **Email**: prashantmanandhar2002@gmail.com
- **GitHub**: [Eemayas](https://github.com/Eemayas)
- **Website**:[https://www.manandharprashant.com.np/](https://www.manandharprashant.com.np/)
