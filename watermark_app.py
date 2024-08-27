import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
from io import BytesIO

# Set up the Streamlit page layout
st.set_page_config(layout="wide")

# Function to apply SVD and modify singular values


def embed_watermark(hr_channel, wm_channel, alpha=0.6):
    """
    Embeds a watermark into the host image channel using SVD.

    Args:
        hr_channel (numpy.ndarray): The host image channel.
        wm_channel (numpy.ndarray): The watermark image channel.
        alpha (float): Scaling factor for watermark embedding.

    Returns:
        numpy.ndarray: The host image channel with embedded watermark.
    """
    U_hr, sigma_hr, V_hr = np.linalg.svd(hr_channel, full_matrices=False)
    U_wm, sigma_wm, V_wm = np.linalg.svd(wm_channel, full_matrices=False)
    sigma_hr_mod = sigma_hr + alpha * sigma_wm
    hr_channel_mod = np.dot(U_hr, np.dot(np.diag(sigma_hr_mod), V_hr))
    return hr_channel_mod

# Function to extract the watermark


def extract_watermark(hr_mod_channel, hr_channel, wr_channel, alpha=0.6):
    """
    Extracts a watermark from a watermarked image channel using SVD.

    Args:
        hr_mod_channel (numpy.ndarray): The watermarked image channel.
        hr_channel (numpy.ndarray): The original host image channel.
        wr_channel (numpy.ndarray): The original watermark image channel.
        alpha (float): Scaling factor used during embedding.

    Returns:
        numpy.ndarray: The extracted watermark channel.
    """
    U_hr_mod, sigma_hr_mod, V_hr_mod = np.linalg.svd(
        hr_mod_channel, full_matrices=False)
    U_hr, sigma_hr, V_hr = np.linalg.svd(hr_channel, full_matrices=False)
    U_wm, sigma_wm, V_wm = np.linalg.svd(wr_channel, full_matrices=False)
    sigma_wm_mod = (sigma_hr_mod - sigma_hr) / alpha
    wm_channel_mod = np.dot(U_wm, np.dot(np.diag(sigma_wm_mod), V_wm))
    return wm_channel_mod


# Streamlit app title
st.title("Watermark Embedding and Extraction")

# Toggle for Embedding or Extraction
operation = st.selectbox("Select Operation", ["Embedding", "Extraction"])

if operation == "Embedding":
    # User inputs for host and watermark images
    col1, col2 = st.columns(2)
    with col1:
        host_image_file = st.file_uploader(
            "Upload the Host Image", type=["jpg", "png", "jpeg"])
        if host_image_file:
            st.image(host_image_file, caption="Host Image",
                     use_column_width=True)

    with col2:
        watermark_image_file = st.file_uploader(
            "Upload the Watermark Image", type=["jpg", "png", "jpeg"])
        if watermark_image_file:
            st.image(watermark_image_file, caption="Watermark Image",
                     use_column_width=True)

    if host_image_file and watermark_image_file:
        progress = st.progress(0)
        status_text = st.text("Starting...")

        # Load images
        status_text.text("Loading images...")
        hr_image = Image.open(host_image_file)
        wm_image = Image.open(watermark_image_file)
        progress.progress(2)
        time.sleep(0.5)

        # Resize watermark image to the size of the host image
        status_text.text("Resizing watermark image...")
        wm_image_resized = wm_image.resize(hr_image.size)
        progress.progress(5)
        time.sleep(0.5)

        # Convert to RGB and split channels
        status_text.text("Splitting image channels...")
        hr_r, hr_g, hr_b = hr_image.convert("RGB").split()
        wm_r, wm_g, wm_b = wm_image_resized.convert("RGB").split()
        progress.progress(10)
        time.sleep(0.5)

        # Convert channels to numpy arrays
        status_text.text("Converting image channels to arrays...")
        hr_r = np.array(hr_r)
        hr_g = np.array(hr_g)
        hr_b = np.array(hr_b)
        wm_r = np.array(wm_r)
        wm_g = np.array(wm_g)
        wm_b = np.array(wm_b)
        progress.progress(15)
        time.sleep(0.5)

        # Embed watermark into each channel
        status_text.text("Embedding watermark into red channel...")
        hr_r_mod = embed_watermark(hr_r, wm_r)
        progress.progress(40)
        time.sleep(0.5)

        status_text.text("Embedding watermark into green channel...")
        hr_g_mod = embed_watermark(hr_g, wm_g)
        progress.progress(65)
        time.sleep(0.5)

        status_text.text("Embedding watermark into blue channel...")
        hr_b_mod = embed_watermark(hr_b, wm_b)
        progress.progress(90)
        time.sleep(0.5)

        # Normalize and convert to uint8
        status_text.text("Normalizing modified channels...")
        hr_r_mod = cv2.normalize(
            hr_r_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hr_g_mod = cv2.normalize(
            hr_g_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        hr_b_mod = cv2.normalize(
            hr_b_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        progress.progress(95)
        time.sleep(0.5)

        # Merge modified channels back into an RGB image
        status_text.text(
            "Merging modified channels into the watermarked image...")
        watermarked_image = Image.merge("RGB", (Image.fromarray(
            hr_r_mod), Image.fromarray(hr_g_mod), Image.fromarray(hr_b_mod)))
        progress.progress(100)
        time.sleep(0.5)

        # Display the watermarked image
        status_text.text("Displaying watermarked image")
        st.image(watermarked_image, caption="Watermarked Image",
                 use_column_width=True)

        # Provide download button for watermarked image
        buffer = BytesIO()
        watermarked_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Create columns for center alignment
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="Download Watermarked Image",
                data=buffer,
                file_name="watermarked_image.png",
                mime="image/png"
            )

elif operation == "Extraction":
    # User inputs for host, watermark, and watermarked images
    col1, col2, col3 = st.columns(3)
    with col1:
        host_image_file = st.file_uploader(
            "Upload the Host Image", type=["jpg", "png", "jpeg"])
        if host_image_file:
            st.image(host_image_file, caption="Host Image",
                     use_column_width=True)

    with col2:
        watermark_image_file = st.file_uploader(
            "Upload the Watermark Image", type=["jpg", "png", "jpeg"])
        if watermark_image_file:
            st.image(watermark_image_file, caption="Watermark Image",
                     use_column_width=True)

    with col3:
        watermarked_image_file = st.file_uploader(
            "Upload the Watermarked Image", type=["jpg", "png", "jpeg"])
        if watermarked_image_file:
            st.image(watermarked_image_file,
                     caption="Watermarked Image", use_column_width=True)

    if host_image_file and watermark_image_file and watermarked_image_file:
        progress = st.progress(0)
        status_text = st.text("Starting...")

        # Load images
        status_text.text("Loading images...")
        hr_image = Image.open(host_image_file)
        wm_image = Image.open(watermark_image_file)
        watermarked_image = Image.open(watermarked_image_file)
        progress.progress(2)
        time.sleep(0.5)

        # Resize watermark and watermarked images to the size of the host image
        status_text.text("Resizing images...")
        wm_image_resized = wm_image.resize(hr_image.size)
        watermarked_image_resized = watermarked_image.resize(hr_image.size)
        progress.progress(5)
        time.sleep(0.5)

        # Convert to RGB and split channels
        status_text.text("Splitting image channels...")
        hr_r, hr_g, hr_b = hr_image.convert("RGB").split()
        wm_r, wm_g, wm_b = wm_image_resized.convert("RGB").split()
        hr_r_mod, hr_g_mod, hr_b_mod = watermarked_image_resized.convert(
            "RGB").split()
        progress.progress(10)
        time.sleep(0.5)

        # Convert channels to numpy arrays
        status_text.text("Converting image channels to arrays...")
        hr_r = np.array(hr_r)
        hr_g = np.array(hr_g)
        hr_b = np.array(hr_b)
        wm_r = np.array(wm_r)
        wm_g = np.array(wm_g)
        wm_b = np.array(wm_b)
        hr_r_mod = np.array(hr_r_mod)
        hr_g_mod = np.array(hr_g_mod)
        hr_b_mod = np.array(hr_b_mod)
        progress.progress(15)
        time.sleep(0.5)

        # Extract watermark from each channel
        status_text.text("Extracting watermark from red channel...")
        wm_r_mod = extract_watermark(hr_r_mod, hr_r, wm_r)
        progress.progress(50)
        time.sleep(0.5)

        status_text.text("Extracting watermark from green channel...")
        wm_g_mod = extract_watermark(hr_g_mod, hr_g, wm_g)
        progress.progress(75)
        time.sleep(0.5)

        status_text.text("Extracting watermark from blue channel...")
        wm_b_mod = extract_watermark(hr_b_mod, hr_b, wm_b)
        progress.progress(90)
        time.sleep(0.5)

        # Normalize and convert to uint8
        status_text.text("Normalizing extracted watermark channels...")
        wm_r_mod = cv2.normalize(
            wm_r_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        wm_g_mod = cv2.normalize(
            wm_g_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        wm_b_mod = cv2.normalize(
            wm_b_mod, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        progress.progress(95)
        time.sleep(0.5)

        # Merge extracted channels back into an RGB image
        status_text.text(
            "Merging extracted channels into the final watermark image...")
        extracted_watermark = Image.merge("RGB", (Image.fromarray(
            wm_r_mod), Image.fromarray(wm_g_mod), Image.fromarray(wm_b_mod)))
        progress.progress(100)
        time.sleep(0.5)

        # Display the extracted watermark
        status_text.text("Displaying extracted watermark")
        st.image(extracted_watermark, caption="Extracted Watermark",
                 use_column_width=True)

        # Provide download button for extracted watermark
        buffer = BytesIO()
        extracted_watermark.save(buffer, format="PNG")
        buffer.seek(0)

        # Create columns for center alignment
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="Download Extracted Watermark",
                data=buffer,
                file_name="extracted_watermark.png",
                mime="image/png"
            )
