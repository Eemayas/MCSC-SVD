import numpy as np
import cv2
import pywt

def apply_svd(image):
    """Apply SVD to an image."""
    U, S, V = np.linalg.svd(image, full_matrices=False)
    return U, S, V


def dwt2(image):
    """Apply 2D DWT to an image."""
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, (LH, HL, HH)


def idwt2(LL, coeffs):
    """Apply inverse 2D DWT to reconstruct an image."""
    return pywt.idwt2((LL, coeffs), 'haar')


def embed_watermark(host_image, watermark_image, alpha=0.1):
    """Embed a watermark into the host image using DWT-SVD."""
    # Apply DWT to both host and watermark images
    host_LL, host_coeffs = dwt2(host_image)
    watermark_LL, _ = dwt2(watermark_image)

    # Apply SVD to the LL sub-band of both images
    U_host, S_host, V_host = apply_svd(host_LL)
    U_wm, S_wm, V_wm = apply_svd(watermark_LL)

    # Embed the watermark
    S_embedded = S_host + alpha * S_wm

    # Reconstruct the LL sub-band using the modified singular values
    LL_embedded = np.dot(U_host, np.dot(np.diag(S_embedded), V_host))

    # Reconstruct the image using inverse DWT
    watermarked_image = idwt2(LL_embedded, host_coeffs)

    return np.clip(watermarked_image, 0, 255).astype(np.uint8)


def extract_watermark(host_image, watermarked_image, alpha=0.1):
    """Extract the watermark from the watermarked image using DWT-SVD."""
    # Apply DWT to both host and watermarked images
    host_LL, _ = dwt2(host_image)
    watermarked_LL, _ = dwt2(watermarked_image)

    # Apply SVD to the LL sub-band of both images
    U_host, S_host, V_host = apply_svd(host_LL)
    U_wm, S_wm, V_wm = apply_svd(watermarked_LL)

    # Ensure dimensions match before proceeding
    min_len = min(len(S_host), len(S_wm))
    S_host = S_host[:min_len]
    S_wm = S_wm[:min_len]

    # Extract the watermark's singular values
    S_extracted = (S_wm - S_host) / alpha

    # Reconstruct the watermark LL sub-band using the extracted singular values
    watermark_LL = np.dot(U_wm, np.dot(np.diag(S_extracted), V_wm))

    # Normalize the watermark to an appropriate range
    watermark_LL = np.clip(watermark_LL, 0, 255)

    return watermark_LL.astype(np.uint8)


# Load images
host_image = cv2.imread('host_image.jpg', cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.imread('watermark_image.jpg', cv2.IMREAD_GRAYSCALE)

# Resize watermark to match the host image
watermark_image = cv2.resize(
    watermark_image, (host_image.shape[1], host_image.shape[0]))

# Embed watermark
watermarked_image = embed_watermark(host_image, watermark_image)

# Save watermarked image
cv2.imwrite('watermarked_image.png', watermarked_image)

# Extract watermark
extracted_watermark = extract_watermark(host_image, watermarked_image)
watermarked_image_grayscale = Image.fromarray(hr_mod)
watermarked_image_grayscale.show()
# Save extracted watermark
cv2.imwrite('extracted_watermark.png', extracted_watermark)
