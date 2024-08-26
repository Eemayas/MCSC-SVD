# import numpy as np
# import pywt
# from skimage import io, color
# from scipy.linalg import svd
# from skimage.transform import resize
# import matplotlib.pyplot as plt


# class Logic:
#     def __init__(self):
#         self.SourceFile = None
#         self.SourceImage = None
#         self.WMFile = None
#         self.WMImage = None
#         self.Psize = None
#         self.Alpha = None
#         self.Sy = None
#         self.Uw = None
#         self.Vw = None
#         self.I_1 = None
#         self.EmbedEnable = False
#         self.WatermarkedImage = None

#     def initializeSource(self, SourceFile, img_type):
#         self.SourceFile = SourceFile
#         self.SourceImage = io.imread(self.SourceFile)

#         if img_type == 'Grayscale' and len(self.SourceImage.shape) == 3:
#             self.SourceImage = color.rgb2gray(self.SourceImage)
#         elif len(self.SourceImage.shape) == 3:
#             if img_type == 'R Component':
#                 self.SourceImage = self.SourceImage[:, :, 0]
#             elif img_type == 'G Component':
#                 self.SourceImage = self.SourceImage[:, :, 1]
#             elif img_type == 'B Component':
#                 self.SourceImage = self.SourceImage[:, :, 2]

#         LL1, _ = pywt.dwt2(self.SourceImage, 'haar')
#         LL2, _ = pywt.dwt2(LL1, 'haar')
#         self.Psize = LL2.shape

#         if self.WMImage is not None:
#             self.EmbedEnable = True

#     def initializeWM(self, WMFile, img_type):
#         self.WMFile = WMFile
#         WMImg = io.imread(self.WMFile)

#         if img_type == 'Grayscale' and len(WMImg.shape) == 3:
#             WMImg = color.rgb2gray(WMImg)
#         elif len(WMImg.shape) == 3:
#             if img_type == 'R Component':
#                 WMImg = WMImg[:, :, 0]
#             elif img_type == 'G Component':
#                 WMImg = WMImg[:, :, 1]
#             elif img_type == 'B Component':
#                 WMImg = WMImg[:, :, 2]

#         self.WMImage = np.array(WMImg)
#         self.WMImage = self.resize(self.WMImage, self.Psize)

#         if self.SourceImage is not None:
#             self.EmbedEnable = True

#     def embed(self, Alpha):
#         self.Alpha = Alpha
#         LL1, (HL1, LH1, HH1) = pywt.dwt2(self.SourceImage, 'haar')
#         LL2, (HL2, LH2, HH2) = pywt.dwt2(LL1, 'haar')
#         Uy, Sy, Vy = svd(LL2, full_matrices=False)
#         self.Sy = Sy
#         self.Uw, Sw, self.Vw = svd(self.WMImage, full_matrices=False)
#         Smark = Sy + Alpha * Sw
#         LL2_1 = np.dot(Uy, np.dot(np.diag(Smark), Vy))
#         LL1_1 = pywt.idwt2((LL2_1, (HL2, LH2, HH2)), 'haar')
#         LL1_1 = self.resize(LL1_1, HL1.shape)
#         self.I_1 = pywt.idwt2((LL1_1, (HL1, LH1, HH1)), 'haar')
#         self.WatermarkedImage = np.uint8(self.I_1)

#         # Save and show watermarked image
#         io.imsave('watermarked_image.png', self.WatermarkedImage)
#         plt.figure(figsize=(6, 6))
#         plt.title('Watermarked Image')
#         plt.imshow(self.WatermarkedImage, cmap='gray')
#         plt.axis('off')
#         plt.show()

#     def extract(self):
#         LL1_wmv, _ = pywt.dwt2(self.I_1, 'haar')
#         LL2_wmv, _ = pywt.dwt2(LL1_wmv, 'haar')
#         _, Sy_wmv, _ = svd(LL2_wmv, full_matrices=False)
#         Swrec = (Sy_wmv - self.Sy) / self.Alpha
#         WMy = np.dot(self.Uw, np.dot(np.diag(Swrec), self.Vw))
#         ExtractedImage = np.uint8(WMy)

#         # Save and show extracted image
#         io.imsave('extracted_image.png', ExtractedImage)
#         plt.figure(figsize=(6, 6))
#         plt.title('Extracted Image')
#         plt.imshow(ExtractedImage, cmap='gray')
#         plt.axis('off')
#         plt.show()

#         return ExtractedImage

#     def resize(self, image, size):
#         return resize(image, size, anti_aliasing=True)


# # Example usage
# logic_instance = Logic()
# logic_instance.initializeSource('host_image.jpg', 'R Component')
# logic_instance.initializeWM('watermark_image.jpg', 'Grayscale')
# logic_instance.embed(0.05)
# extracted_image = logic_instance.extract()

import numpy as np
import pywt
from skimage import io, color
from scipy.linalg import svd
from skimage.transform import resize
import matplotlib.pyplot as plt


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

    LL1, _ = pywt.dwt2(source_image, 'haar')
    LL2, _ = pywt.dwt2(LL1, 'haar')
    psize = LL2.shape

    return source_image, psize


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


def embed(source_image, wm_image, alpha):
    LL1, (HL1, LH1, HH1) = pywt.dwt2(source_image, 'haar')
    LL2, (HL2, LH2, HH2) = pywt.dwt2(LL1, 'haar')
    Uy, Sy, Vy = svd(LL2, full_matrices=False)
    Uw, Sw, Vw = svd(wm_image, full_matrices=False)
    Smark = Sy + alpha * Sw
    LL2_1 = np.dot(Uy, np.dot(np.diag(Smark), Vy))
    LL1_1 = pywt.idwt2((LL2_1, (HL2, LH2, HH2)), 'haar')
    LL1_1 = resize(LL1_1, HL1.shape, anti_aliasing=True)
    I_1 = pywt.idwt2((LL1_1, (HL1, LH1, HH1)), 'haar')
    watermarked_image = np.uint8(I_1)

    # Save and show watermarked image
    io.imsave('watermarked_image.png', watermarked_image)
    plt.figure(figsize=(6, 6))
    plt.title('Watermarked Image')
    plt.imshow(watermarked_image, cmap='gray')
    plt.axis('off')
    plt.show()

    return watermarked_image, Sy, Uw, Vw


def extract(watermarked_image, Sy, Uw, Vw, alpha):
    LL1_wmv, _ = pywt.dwt2(watermarked_image, 'haar')
    LL2_wmv, _ = pywt.dwt2(LL1_wmv, 'haar')
    _, Sy_wmv, _ = svd(LL2_wmv, full_matrices=False)
    Swrec = (Sy_wmv - Sy) / alpha
    extracted_image = np.dot(Uw, np.dot(np.diag(Swrec), Vw))
    extracted_image = np.uint8(extracted_image)

    # Save and show extracted image
    io.imsave('extracted_image.png', extracted_image)
    plt.figure(figsize=(6, 6))
    plt.title('Extracted Image')
    plt.imshow(extracted_image, cmap='gray')
    plt.axis('off')
    plt.show()

    return extracted_image


# Example usage
source_img, psize = initialize_source('host_image.jpg', 'R Component')
wm_img = initialize_wm('watermark_image.jpg', 'Grayscale', psize)
wmd_img=initialize_wm('watermarked_image.png', 'Grayscale', psize)
# watermarked_img, Sy, Uw, Vw = embed(source_img, wm_img, 0.05)
# extracted_img = extract(watermarked_img, Sy, Uw, Vw, 0.05)
extracted_img = extract(wmd_img, Sy, Uw, Vw, 0.05)
