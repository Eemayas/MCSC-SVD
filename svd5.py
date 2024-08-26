# %% [markdown]
# # Watermark Extraction

# %%
from PIL import Image
import numpy as np
import cv2

# %% [markdown]
# ### Load the original host image and watermarked image

# %%
original_image = Image.open("host_image.jpg")
watermarked_image = Image.open("output_image-1.png")

# Convert images to RGB (if not already in that mode)
original_image = original_image.convert("RGB")
watermarked_image = watermarked_image.convert("RGB")

# Split images into R, G, B channels
orig_r, orig_g, orig_b = original_image.split()
wm_r, wm_g, wm_b = watermarked_image.split()

# Convert channels to numpy arrays
orig_r = np.array(orig_r)
orig_g = np.array(orig_g)
orig_b = np.array(orig_b)

wm_r = np.array(wm_r)
wm_g = np.array(wm_g)
wm_b = np.array(wm_b)

# %% [markdown]
# ### Function to extract watermark from each channel

# %%


def extract_watermark(orig_channel, wm_channel, alpha=0.6):
    U_orig, sigma_orig, V_orig = np.linalg.svd(orig_channel)
    U_wm, sigma_wm, V_wm = np.linalg.svd(wm_channel)

    # Extract the watermark
    sigma_extracted = (sigma_wm - sigma_orig) / alpha

    # Reconstruct the watermark channel
    extracted_channel = np.dot(U_wm[:, :sigma_extracted.shape[0]], np.dot(
        np.diag(sigma_extracted), V_wm[:sigma_extracted.shape[0], :]))

    return extracted_channel

# %% [markdown]
# ### Extract watermark from each channel


# %%
extracted_r = extract_watermark(orig_r, wm_r)
extracted_g = extract_watermark(orig_g, wm_g)
extracted_b = extract_watermark(orig_b, wm_b)

# %%
# Normalize and convert to uint8
extracted_r = cv2.normalize(
    extracted_r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
extracted_g = cv2.normalize(
    extracted_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
extracted_b = cv2.normalize(
    extracted_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# %%
# Merge extracted channels back into an RGB image
extracted_watermark = Image.merge("RGB", (Image.fromarray(
    extracted_r), Image.fromarray(extracted_g), Image.fromarray(extracted_b)))
extracted_watermark.save('extracted_watermark.png')
extracted_watermark.show()

# %% [markdown]
# ### Post-processing to enhance the extracted watermark (optional)

# %%


def enhance_watermark(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert to LAB color space
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to RGB color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


# %%
# Enhance the extracted watermark
enhanced_watermark = enhance_watermark(extracted_watermark)
enhanced_watermark.save('enhanced_extracted_watermark.png')
enhanced_watermark.show()
