from PIL import Image
import numpy as np
import cv2

# Load the host and watermark images
hr_image = Image.open("host_image.jpg")
wm_image = Image.open("watermark_image.jpg")

# Resize the watermark image to the same size as the host image
wm_image_resized = wm_image.resize(hr_image.size)

# Convert images to RGB (if not already in that mode)
hr_image = hr_image.convert("RGB")
wm_image_resized = wm_image_resized.convert("RGB")

# Split images into R, G, B channels
hr_r, hr_g, hr_b = hr_image.split()
wm_r, wm_g, wm_b = wm_image_resized.split()

# Convert channels to numpy arrays
hr_r = np.array(hr_r)
hr_g = np.array(hr_g)
hr_b = np.array(hr_b)

wm_r = np.array(wm_r)
wm_g = np.array(wm_g)
wm_b = np.array(wm_b)

# Function to apply SVD and modify singular values
def embed_watermark(hr_channel, wm_channel, alpha=0.05):
    U_hr, sigma_hr, V_hr = np.linalg.svd(hr_channel, full_matrices=False)
    U_wm, sigma_wm, V_wm = np.linalg.svd(wm_channel, full_matrices=False)
    
    # Embed the watermark by modifying the singular values
    sigma_hr_mod = sigma_hr + alpha * sigma_wm
    hr_channel_mod = np.dot(U_hr, np.dot(np.diag(sigma_hr_mod), V_hr))
    
    return hr_channel_mod, sigma_hr, sigma_hr_mod

# Embed watermark in each channel
hr_r_mod, sigma_hr_r, sigma_hr_mod_r = embed_watermark(hr_r, wm_r)
hr_g_mod, sigma_hr_g, sigma_hr_mod_g = embed_watermark(hr_g, wm_g)
hr_b_mod, sigma_hr_b, sigma_hr_mod_b = embed_watermark(hr_b, wm_b)

# Normalize and convert to uint8
hr_r_mod = cv2.normalize(hr_r_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hr_g_mod = cv2.normalize(hr_g_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hr_b_mod = cv2.normalize(hr_b_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Merge modified channels back into an RGB image
watermarked_image = Image.merge("RGB", (Image.fromarray(hr_r_mod), Image.fromarray(hr_g_mod), Image.fromarray(hr_b_mod)))
watermarked_image.save('output_image.png')
watermarked_image.show()

# Extraction process
def extract_watermark(hr_mod_channel, hr_channel, sigma_hr, alpha=0.05):
    U_hr_mod, sigma_hr_mod, V_hr_mod = np.linalg.svd(hr_mod_channel, full_matrices=False)
    
    # Extract the watermark singular values
    sigma_wm_mod = (sigma_hr_mod - sigma_hr) / alpha
    wm_channel_mod = np.dot(U_hr_mod, np.dot(np.diag(sigma_wm_mod), V_hr_mod))
    
    return wm_channel_mod

# Extract watermark from each channel
wm_r_mod = extract_watermark(hr_r_mod, hr_r, sigma_hr_r)
wm_g_mod = extract_watermark(hr_g_mod, hr_g, sigma_hr_g)
wm_b_mod = extract_watermark(hr_b_mod, hr_b, sigma_hr_b)

# Normalize and convert to uint8
wm_r_mod = cv2.normalize(wm_r_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
wm_g_mod = cv2.normalize(wm_g_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
wm_b_mod = cv2.normalize(wm_b_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Merge extracted channels back into an RGB image
extracted_watermark = Image.merge("RGB", (Image.fromarray(wm_r_mod), Image.fromarray(wm_g_mod), Image.fromarray(wm_b_mod)))
extracted_watermark.save('watermark_output_image.png')
extracted_watermark.show()
