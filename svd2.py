import cv2
import pywt
import numpy as np


def diag(s, shape_LL):
    """
    To recover the singular values to be a matrix.
    :param s: a 1D numpy array
    :param shape_LL: shape of the LL band from DWT
    :return: S matrix
    """
    S = np.zeros(shape_LL)
    row = min(S.shape)
    S[:row, :row] = np.diag(s)
    return S


def calculate(img, wavelet, level):
    """
    To calculate the DWT Coefficients and SVD components.
    :param img: should be a numpy array or the path of the image.
    :param wavelet: type of wavelet used for DWT
    :param level: level of decomposition
    :return: DWT coefficients and SVD components (U, S, V)
    """
    Coefficients = pywt.wavedec2(img, wavelet=wavelet, level=level)
    U, S, V = np.linalg.svd(Coefficients[0])
    return Coefficients, U, S, V


def recover(Coefficients, U, S, V, wavelet):
    """
    To recover the image from the SVD components and DWT coefficients.
    :param Coefficients: DWT coefficients
    :param U: U matrix from SVD
    :param S: singular values matrix
    :param V: V matrix from SVD
    :param wavelet: type of wavelet used for DWT
    :return: Reconstructed image
    """
    Coefficients[0] = U.dot(S).dot(V)
    return pywt.waverec2(Coefficients, wavelet=wavelet)


def embed_watermark(img_path, watermark_path, wavelet="haar", level=2, ratio=0.1, path_save=None):
    """
    Embed a watermark into an image using DWT-SVD.
    :param img_path: Path to the input image
    :param watermark_path: Path to the watermark image
    :param wavelet: Type of wavelet used for DWT
    :param level: Level of decomposition for DWT
    :param ratio: Embedding strength
    :param path_save: Path to save the watermarked image
    """
    if not path_save:
        path_save = "watermarked_" + img_path
    if isinstance(img_path, str):
        host_img = cv2.imread(img_path, 0)

    if isinstance(watermark_path, str):
        watermark_img = cv2.imread(watermark_path, 0)

    # Resize the watermark image to the size of the host image
    watermark_img_resized = cv2.resize(
        watermark_img, (host_img.shape[1], host_img.shape[0]))

    # Calculate SVD components for the input image
    img_coeff, img_U, img_S, img_V = calculate(host_img, wavelet, level)
    shape_LL = img_coeff[0].shape

    # Calculate SVD components for the watermark image
    watermark_coeff, watermark_U, watermark_S, watermark_V = calculate(
        watermark_img_resized, wavelet, level)

    # Embed the watermark into the image
    S_img = img_S + ratio * watermark_S * (img_S.max() / watermark_S.max())
    img_recovered = recover(img_coeff, img_U, diag(
        S_img, shape_LL), img_V, wavelet)

    cv2.imwrite(path_save, img_recovered)
    print(f"Watermarked image saved to: {path_save}")




def extract_watermark(watermarked_img_path, original_img_path, wavelet="haar", level=2, ratio=0.1, extracted_watermark_path=None):
    """
    Extract the watermark from a watermarked image using DWT-SVD.
    :param watermarked_img_path: Path to the watermarked image
    :param original_img_path: Path to the original image used for embedding
    :param wavelet: Type of wavelet used for DWT
    :param level: Level of decomposition for DWT
    :param ratio: Embedding strength used during watermarking
    :param extracted_watermark_path: Path to save the extracted watermark image
    """
    if not extracted_watermark_path:
        extracted_watermark_path = "extracted_watermark_" + watermarked_img_path
    if isinstance(watermarked_img_path, str):
        watermarked_img = cv2.imread(watermarked_img_path, 0)

    if isinstance(original_img_path, str):
        original_img = cv2.imread(original_img_path, 0)
    # Calculate SVD components for the watermarked image
    watermarked_coeff, watermarked_U, watermarked_S, watermarked_V = calculate(
        watermarked_img, wavelet, level)

    # Calculate SVD components for the original image
    original_coeff, original_U, original_S, original_V = calculate(
        original_img, wavelet, level)

    # Extract the watermark
    S_watermark = (watermarked_S - original_S) / ratio
    shape_LL = original_coeff[0].shape
    watermark_recovered = recover(original_coeff, original_U, diag(
        S_watermark, shape_LL), original_V, wavelet)

    cv2.imwrite(extracted_watermark_path, watermark_recovered)
    print(f"Extracted watermark saved to: {extracted_watermark_path}")

if __name__ == '__main__':
    # embed_watermark("host_image.jpg", "watermark_image.jpg", level=3)

    extract_watermark("watermarked_host_image.jpg", "host_image.jpg", level=3)
