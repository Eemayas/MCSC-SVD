import numpy as np
import pywt
from skimage import io, color
from scipy.linalg import svd
from skimage.transform import resize
import matplotlib.pyplot as plt

# Function to initialize the source image
def initialize_source(source_file, img_type):
    source_image = io.imread(source_file)

    if img_type == 'Grayscale' and len(source_image.shape) == 3:
        source_image = color.rgb2gray(source_image)
    elif len(source_image.shape) == 3:
        if img_type == 'R Component':
            source_image = source_image[:, :, 0]
        elif img_type == 'G Component':
            source_image = source_image[:, :, 1]
        elif img_type == 'B Component':
            source_image = source_image[:, :, 2]

    # Perform two-level DWT for image size calculation
    LL1, _ = pywt.dwt2(source_image, 'haar')
    LL2, _ = pywt.dwt2(LL1, 'haar')
    psize = LL2.shape

    return source_image, psize

# Function to initialize the watermark image
def initialize_wm(wm_file, img_type, psize):
    wm_image = io.imread(wm_file)

    if img_type == 'Grayscale' and len(wm_image.shape) == 3:
        wm_image = color.rgb2gray(wm_image)
    elif len(wm_image.shape) == 3:
        if img_type == 'R Component':
            wm_image = wm_image[:, :, 0]
        elif img_type == 'G Component':
            wm_image = wm_image[:, :, 1]
        elif img_type == 'B Component':
            wm_image = wm_image[:, :, 2]

    wm_image = resize(wm_image, psize, anti_aliasing=True)

    return np.array(wm_image)

# Function to embed the watermark in one channel
def embed_channel(source_channel, wm_channel, alpha):
    LL1, (HL1, LH1, HH1) = pywt.dwt2(source_channel, 'haar')
    LL2, (HL2, LH2, HH2) = pywt.dwt2(LL1, 'haar')
    Uy, Sy, Vy = svd(LL2, full_matrices=False)
    Uw, Sw, Vw = svd(wm_channel, full_matrices=False)
    Smark = Sy + alpha * Sw
    LL2_1 = np.dot(Uy, np.dot(np.diag(Smark), Vy))
    LL1_1 = pywt.idwt2((LL2_1, (HL2, LH2, HH2)), 'haar')
    LL1_1 = resize(LL1_1, HL1.shape, anti_aliasing=True)
    I_1 = pywt.idwt2((LL1_1, (HL1, LH1, HH1)), 'haar')

    return np.uint8(I_1), Sy, Uw, Vw

# Function to extract the watermark from one channel
def extract_channel(watermarked_channel, Sy, Uw, Vw, alpha):
    LL1_wmv, _ = pywt.dwt2(watermarked_channel, 'haar')
    LL2_wmv, _ = pywt.dwt2(LL1_wmv, 'haar')
    _, Sy_wmv, _ = svd(LL2_wmv, full_matrices=False)
    Swrec = (Sy_wmv - Sy) / alpha
    extracted_channel = np.dot(Uw, np.dot(np.diag(Swrec), Vw))

    return np.uint8(extracted_channel)

# Main embedding function
def embed(source_image, wm_image, alpha, img_type):
    if len(source_image.shape) == 2:  # Grayscale
        watermarked_image, Sy, Uw, Vw = embed_channel(source_image, wm_image, alpha)
        return watermarked_image, [Sy], [Uw], [Vw]
    else:  # Color image
        watermarked_channels = []
        S_data, U_data, V_data = [], [], []
        for i in range(3):  # Process R, G, B channels separately
            watermarked_channel, Sy, Uw, Vw = embed_channel(source_image[:, :, i], wm_image[:, :, i], alpha)
            watermarked_channels.append(watermarked_channel)
            S_data.append(Sy)
            U_data.append(Uw)
            V_data.append(Vw)
        watermarked_image = np.stack(watermarked_channels, axis=2)
        return watermarked_image, S_data, U_data, V_data

# Main extraction function
def extract(watermarked_image, S_data, U_data, V_data, alpha, img_type):
    if len(watermarked_image.shape) == 2:  # Grayscale
        extracted_image = extract_channel(watermarked_image, S_data[0], U_data[0], V_data[0], alpha)
        return extracted_image
    else:  # Color image
        extracted_channels = []
        for i in range(3):  # Process R, G, B channels separately
            extracted_channel = extract_channel(watermarked_image[:, :, i], S_data[i], U_data[i], V_data[i], alpha)
            extracted_channels.append(extracted_channel)
        extracted_image = np.stack(extracted_channels, axis=2)
        return extracted_image

# Example usage
source_img, psize = initialize_source('host_image.jpg', 'Color')
wm_img = initialize_wm('watermark_image.jpg', 'Color', psize)
watermarked_img, S_data, U_data, V_data = embed(source_img, wm_img, 0.05, 'Color')
extracted_img = extract(watermarked_img, S_data, U_data, V_data, 0.05, 'Color')

# Save and show results
io.imsave('watermarked_image.png', watermarked_img)
io.imsave('extracted_image.png', extracted_img)
plt.figure(figsize=(6, 6))
plt.title('Watermarked Image')
plt.imshow(watermarked_img)
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.title('Extracted Image')
plt.imshow(extracted_img)
plt.axis('off')
plt.show()
