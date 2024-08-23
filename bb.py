from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def embed_watermark(host_image_path, watermark_image_path, output_image_path, alpha=0.6):
    # Load the host and watermark images
    hr_image = Image.open(host_image_path).convert("RGB")
    wm_image = Image.open(watermark_image_path).convert("RGB")

    # Resize the watermark image to the same size as the host image
    wm_image_resized = wm_image.resize(hr_image.size)

    # Convert images to numpy arrays
    hr_image_np = np.array(hr_image)
    wm_image_np = np.array(wm_image_resized)

    # Prepare an empty array to store the modified image
    hr_image_mod_np = np.zeros_like(hr_image_np)

    # Process each color channel with a progress bar
    for channel in tqdm(range(3), desc="Embedding Watermark"):  # For R, G, B channels
        try:
            # Calculate SVD of the host image
            U_hr, sigma_hr, V_hr = np.linalg.svd(hr_image_np[:, :, channel])

            # Calculate SVD of the watermark image
            U_wm, sigma_wm, V_wm = np.linalg.svd(wm_image_np[:, :, channel])

            # Modify the S matrix of the host image
            sigma_hr_mod = sigma_hr + alpha * sigma_wm

            # Inverse SVD to obtain the watermarked image for the current channel
            hr_image_mod_np[:, :, channel] = np.dot(U_hr[:, :sigma_hr_mod.shape[0]],
                                                    np.dot(np.diag(sigma_hr_mod),
                                                           V_hr[:sigma_hr_mod.shape[0], :]))
        except Exception as e:
            print(f"Error processing channel {channel}: {e}")
            return

    # Normalize the array to the range of 0 to 255
    hr_image_mod_np = cv2.normalize(hr_image_mod_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create the final watermarked image
    hr_image_mod = Image.fromarray(hr_image_mod_np)
    hr_image_mod.save(output_image_path)

def extract_watermark(watermarked_image_path, host_image_path, extracted_watermark_path, alpha=0.6):
    # Load the images
    hr_image_mod = Image.open(watermarked_image_path).convert("RGB")
    hr_image = Image.open(host_image_path).convert("RGB")

    # Convert images to numpy arrays
    hr_image_mod_np = np.array(hr_image_mod)
    hr_image_np = np.array(hr_image)

    # Prepare an empty array to store the extracted watermark
    watermark_mod_np = np.zeros_like(hr_image_np)

    # Process each color channel with a progress bar
    for channel in tqdm(range(3), desc="Extracting Watermark"):  # For R, G, B channels
        try:
            # Calculate SVD of the watermarked image
            U_hr_mod1, sigma_hr_mod1, V_hr_mod1 = np.linalg.svd(hr_image_mod_np[:, :, channel])

            # Calculate SVD of the original host image
            U_hr1, sigma_hr1, V_hr1 = np.linalg.svd(hr_image_np[:, :, channel])

            # Find the original S matrix for the watermark
            sigma_wm_mod = (sigma_hr_mod1 - sigma_hr1) / alpha

            # Reconstruct the watermark for the current channel
            watermark_mod_np[:, :, channel] = np.dot(U_hr1[:, :sigma_wm_mod.shape[0]],
                                                     np.dot(np.diag(sigma_wm_mod),
                                                            V_hr1[:sigma_wm_mod.shape[0], :]))
        except Exception as e:
            print(f"Error processing channel {channel}: {e}")
            return

    # Normalize the extracted watermark to the range of 0 to 255
    watermark_mod_np = cv2.normalize(watermark_mod_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create the final extracted watermark image
    extracted_watermark = Image.fromarray(watermark_mod_np)
    extracted_watermark.save(extracted_watermark_path)

if __name__ == "__main__":
    # Paths to images
    host_image_path = "host_image.jpg"
    watermark_image_path = "watermark_image.jpg"
    output_image_path = "output_image1.png"
    extracted_watermark_path = "extracted_watermark1.png"

    # Embed watermark
    embed_watermark(host_image_path, watermark_image_path, output_image_path)

    # Extract watermark
    extract_watermark(output_image_path, host_image_path, extracted_watermark_path)

    print("Watermark embedding and extraction completed.")
