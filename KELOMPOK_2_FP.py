import streamlit as st
import imageio
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure, img_as_ubyte
import scipy.ndimage as ndi
from skimage import filters, morphology 
import math
import pandas as pd
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.io import imread

image_segmented1 = None
image_segmented2 = None

# Fungsi untuk mengonversi gambar menjadi grayscale
def convert_to_grayscale(image):
    # Menggunakan AHE untuk meningkatkan kontras
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.01)
    img_adapteq = img_as_ubyte(img_adapteq)  # Convert to uint8

    # Menghitung grayscale dari gambar yang sudah di AHE
    my_gray = img_adapteq @ np.array([0.2126, 0.7152, 0.0722])
    return my_gray

# Fungsi untuk memproses AHE pada gambar
def ahe_processing(image):
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.01)
    img_adapteq = img_as_ubyte(img_adapteq)
    return img_adapteq

# Fungsi untuk menampilkan histogram gambar
def plot_histogram(image, title, x_range=(0, 255), y_range=(0, 15000)):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(image.ravel(), bins=256, color='orange', alpha=0.5, label='Total')
    if len(image.shape) == 3:  # Jika gambar berwarna (RGB)
        ax.hist(image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
        ax.hist(image[:, :, 1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
        ax.hist(image[:, :, 2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(title)
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

# Fungsi untuk Median Filter
def MedianFilter(image, size=3):
    return ndi.median_filter(image, size=size)
def normalize_image(image):
    """Normalisasi gambar agar berada dalam rentang [0, 255] dan bertipe uint8."""
    image_normalized = (image - image.min()) / (image.max() - image.min())  # Normalisasi ke [0, 1]
    image_normalized = (image_normalized * 255).astype(np.uint8)  # Skala ke [0, 255] dan ubah ke uint8
    return image_normalized

# Memuat gambar
uploaded_file1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

# --- Bagian Streamlit ---
st.title("Image Processing Dashboard")

# Sidebar menu untuk navigasi
menu = st.sidebar.radio("Menu", ["Home", "Image Details", "Histograms", "Grayscale", "AHE Results", "Median Filter", "Thresholding dan Median Filtering", "Penghitungan dan Visualisasi Histogram Gambar yang Difilter", "region"])

# If images are uploaded, open them
if uploaded_file1 is not None:
    im1 = Image.open(uploaded_file1)
else:
    im1 = None

if uploaded_file2 is not None:
    im2 = Image.open(uploaded_file2)
else:
    im2 = None

# Bagian Home
if menu == "Home":
    st.write("## Welcome to the Image Processing Dashboard")
    if im1 is not None:
        st.image(im1, caption='Image 1')
    else:
        st.write("No Image 1 uploaded.")
    if im2 is not None:
        st.image(im2, caption='Image 2')
    else:
        st.write("No Image 2 uploaded.")

# Detail Gambar
elif menu == "Image Details":
    st.write("## Image Details")
    
    if im1 is not None and im2 is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Details for Image 1:")
            im1_np = np.array(im1)
            st.write(f"Type: {type(im1_np)}")
            st.write(f"dtype: {im1_np.dtype}")
            st.write(f"shape: {im1_np.shape}")
            st.write(f"size: {im1_np.size}")
            st.image(im1, caption="Image 1", use_column_width=True)

        with col2:
            st.write("### Details for Image 2:")
            im2_np = np.array(im2)
            st.write(f"Type: {type(im2_np)}")
            st.write(f"dtype: {im2_np.dtype}")
            st.write(f"shape: {im2_np.shape}")
            st.write(f"size: {im2_np.size}")
            st.image(im2, caption="Image 2", use_column_width=True)
    
    else:
        st.write("Please upload both images to see the details.")

# Histogram Gambar
elif menu == "Histograms":
    st.write("## Histograms for Images")
    if im1 is not None and im2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Histogram for Image 1")
            plot_histogram(im1, 'Histogram for Image 1 (im1)', (0, 255), (0, 15000))
        with col2:
            st.write("### Histogram for Image 2")
            plot_histogram(im2, 'Histogram for Image 2 (im2)', (0, 255), (0, 15000))
    else:
        st.write("Please upload both images to see the histograms.")

# Bagian Grayscale
elif menu == "Grayscale":
    st.write("## Grayscale Conversion")
    if im1 is not None and im2 is not None:
        my_gray1 = convert_to_grayscale(im1)
        my_gray2 = convert_to_grayscale(im2)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Grayscale Image 1")
            st.image(my_gray1.astype(np.uint8), caption="Grayscale Image 1", use_column_width=True)
        with col2:
            st.write("### Grayscale Image 2")
            st.image(my_gray2.astype(np.uint8), caption="Grayscale Image 2", use_column_width=True)
    else:
        st.write("Please upload both images to convert to grayscale.")

# Bagian Hasil AHE
elif menu == "AHE Results":
    st.write("## Adaptive Histogram Equalization (AHE) Results")
    if 'my_gray1' not in locals() or 'my_gray2' not in locals():
        my_gray1 = convert_to_grayscale(im1)
        my_gray2 = convert_to_grayscale(im2)

    if im1 is not None and im2 is not None:
        img_adapteq1 = exposure.equalize_adapthist(my_gray1, clip_limit=0.01)
        img_adapteq2 = exposure.equalize_adapthist(my_gray2, clip_limit=0.01)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### AHE Image 1")
            st.image(img_adapteq1, caption='AHE Image 1', use_column_width=True)
        with col2:
            st.write("### AHE Image 2")
            st.image(img_adapteq2, caption='AHE Image 2', use_column_width=True)
    else:
        st.write("Please upload both images to apply AHE.")

# Bagian Median Filter
elif menu == "Median Filter":
    st.write("## Median Filter Application")

    if 'img_adapteq1' not in locals() or 'img_adapteq2' not in locals():
        my_gray1 = convert_to_grayscale(im1)
        my_gray2 = convert_to_grayscale(im2)
        img_adapteq1 = exposure.equalize_adapthist(my_gray1, clip_limit=0.01)
        img_adapteq2 = exposure.equalize_adapthist(my_gray2, clip_limit=0.01)
        med1 = MedianFilter(img_adapteq1)
        med2 = MedianFilter(img_adapteq2)

    if im1 is not None and im2 is not None:
        med1 = MedianFilter(img_adapteq1)
        med2 = MedianFilter(img_adapteq2)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Median Filtered Image 1")
            st.image(med1.astype(np.uint8), caption="Median Filtered Image 1", use_column_width=True)
        with col2:
            st.write("### Median Filtered Image 2")
            st.image(med2.astype(np.uint8), caption="Median Filtered Image 2", use_column_width=True)
    else:
        st.write("Please upload both images to apply median filter.")
    

# Bagian Thresholding
elif menu == "Thresholding":
    st.write("## Thresholding")
    
    # Pastikan gambar median difilter sudah didefinisikan
    if 'med1' not in locals() or 'med2' not in locals():
        my_gray1 = convert_to_grayscale(im1)
        my_gray2 = convert_to_grayscale(im2)
        med1 = MedianFilter(my_gray1)
        med2 = MedianFilter(my_gray2)
    
    # Cek properti med1 dan med2
    st.write("### Image Properties:")

    if im1 is not None and im2 is not None:
        # Thresholding menggunakan Otsu
        from skimage import filters
        
        threshold1 = filters.threshold_otsu(med1)
        threshold2 = filters.threshold_otsu(med2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Image 1 - Median Filtered:")
            st.write(f"Type: {type(med1)}")
            st.write(f"dtype: {med1.dtype}")
            st.write(f"shape: {med1.shape}")
            st.write(f"size: {med1.size}")
            st.write("Nilai threshold Otsu untuk gambar 1:", threshold1)
            
        with col2:
            st.write("#### Image 2 - Median Filtered:")
            st.write(f"Type: {type(med2)}")
            st.write(f"dtype: {med2.dtype}")
            st.write(f"shape: {med2.shape}")
            st.write(f"size: {med2.size}")
            st.write("Nilai threshold Otsu untuk gambar 2:", threshold2)
        
        # Menampilkan hasil thresholding dan kontur pada Image 1
        st.write("### Thresholding and Contour for Image 1")
        fig1, ax1 = plt.subplots(ncols=2, figsize=(12, 8))
        ax1[0].imshow(med1, cmap='gray')
        ax1[0].contour(med1, [threshold1], colors='purple')
        ax1[0].set_title('Image 1 - Contour at Threshold')
        ax1[1].imshow(med1 < threshold1, cmap='gray')
        ax1[1].set_title('Image 1 - Thresholded')
        st.pyplot(fig1)
        
        # Menampilkan hasil thresholding dan kontur pada Image 2
        st.write("### Thresholding and Contour for Image 2")
        fig2, ax2 = plt.subplots(ncols=2, figsize=(12, 8))
        ax2[0].imshow(med2, cmap='gray')
        ax2[0].contour(med2, [threshold2], colors='purple')
        ax2[0].set_title('Image 2 - Contour at Threshold')
        ax2[1].imshow(med2 < threshold2, cmap='gray')
        ax2[1].set_title('Image 2 - Thresholded')
        st.pyplot(fig2)
        
        # Median Filtering
        st.write("### Median Filtering After Thresholding")
        
        median_filtered1 = filters.median(med1, np.ones((10, 10)))
        median_filtered2 = filters.median(med2, np.ones((10, 10)))
        
        fig3, ax3 = plt.subplots(ncols=2, figsize=(12, 6))
        ax3[0].imshow(median_filtered1, cmap='gray')
        ax3[0].set_title('Median Filtered Image 1')
        ax3[1].imshow(median_filtered2, cmap='gray')
        ax3[1].set_title('Median Filtered Image 2')
        
        st.pyplot(fig3)
    else:
        st.write("Please upload both images to see the thresholding and median filtering results.")

# Bagian Penghitungan dan Visualisasi Histogram Gambar yang Difilter
elif menu == "Penghitungan dan Visualisasi Histogram Gambar yang Difilter":
    st.write("## Histogram of Median Filtered Images")

    # Memastikan gambar yang terfilter sudah ada
    if 'im1' not in locals() or 'im2' not in locals():
        st.error("Please upload both images first.")
    else:
        # Jika gambar sudah ada, lakukan proses Median Filtering terlebih dahulu
        my_gray1 = convert_to_grayscale(im1)
        my_gray2 = convert_to_grayscale(im2)
        med1 = MedianFilter(my_gray1)
        med2 = MedianFilter(my_gray2)
        median_filtered1 = ndi.median_filter(med1, size=10)
        median_filtered2 = ndi.median_filter(med2, size=10)

        # Slice pada median_filtered1
        median_filtered11 = median_filtered1[:, 100:]

        # Normalisasi dan tampilkan gambar median_filtered11 di Streamlit
        st.image(normalize_image(median_filtered11), caption="Median Filtered 11 (Cropped)", use_column_width=True)

        # Thresholding dan contour untuk median_filtered11
        threshold3 = filters.threshold_otsu(median_filtered11)
        fig, ax = plt.subplots()
        ax.imshow(median_filtered11, cmap='gray')
        ax.contour(median_filtered11, [threshold3], colors='red')
        ax.set_title(f'Contour for Median Filtered 11 at Threshold {threshold3}')
        st.pyplot(fig)

        # Thresholding dan contour untuk median_filtered2
        threshold4 = filters.threshold_otsu(median_filtered2)
        fig, ax = plt.subplots()
        ax.imshow(median_filtered2, cmap='gray')
        ax.contour(median_filtered2, [threshold4], colors='red')
        ax.set_title(f'Contour for Median Filtered 2 at Threshold {threshold4}')
        st.pyplot(fig)

        # Binary classification
        binary_image1 = median_filtered11 < threshold3
        st.image(binary_image1.astype(np.uint8) * 255, caption="Binary Classification Image 1", use_column_width=True)

        binary_image2 = median_filtered2 < threshold4
        st.image(binary_image2.astype(np.uint8) * 255, caption="Binary Classification Image 2", use_column_width=True)

        # Remove small objects using morphology
        only_large_blobs1 = morphology.remove_small_objects(binary_image1, min_size=100)
        st.image(only_large_blobs1.astype(np.uint8) * 255, caption="Large Blobs Image 1", use_column_width=True)

        only_large_blobs2 = morphology.remove_small_objects(binary_image2, min_size=100)
        st.image(only_large_blobs2.astype(np.uint8) * 255, caption="Large Blobs Image 2", use_column_width=True)

        # Fill small holes
        only_large1 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs1), min_size=100))
        image_segmented1 = only_large1
        st.image(image_segmented1.astype(np.uint8) * 255, caption="Segmented Image 1", use_column_width=True)

        only_large2 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs2), min_size=100))
        image_segmented2 = only_large2
        st.image(image_segmented2.astype(np.uint8) * 255, caption="Segmented Image 2", use_column_width=True)

        # Calculate histograms for both filtered images
        histo_median1 = ndi.histogram(median_filtered1, min=0, max=255, bins=256)
        histo_median2 = ndi.histogram(median_filtered2, min=0, max=255, bins=256)

        # Create a figure with 2 subplots for side-by-side comparison
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

        # Plot the first histogram
        ax[0].plot(histo_median1, color='blue')
        ax[0].set_title('Histogram of Median Filtered Image 1')
        ax[0].set_xlabel('Pixel Value')
        ax[0].set_ylabel('Frequency')

        # Plot the second histogram
        ax[1].plot(histo_median2, color='green')
        ax[1].set_title('Histogram of Median Filtered Image 2')
        ax[1].set_xlabel('Pixel Value')
        ax[1].set_ylabel('Frequency')

        # Display the plots
        st.pyplot(fig)

# region
elif menu == "region":
    st.write("## Region Analysis of Segmented Images")

    # Memastikan gambar tersegmentasi sudah ada
    if 'image_segmented1' not in locals() or 'image_segmented2' not in locals():
        st.error("Please perform the filtering and segmentation step first.")
    else:
        # Label the segmented images
        label_img1, nlabels1 = ndi.label(image_segmented1)
        label_img2, nlabels2 = ndi.label(image_segmented2)

        # Inform about the number of detected components
        st.write(f"For Image 1, there are {nlabels1} separate components/objects detected.")
        st.write(f"For Image 2, there are {nlabels2} separate components/objects detected.")

        # Process Image 1
        boxes1 = ndi.find_objects(label_img1)
        for label_ind, label_coords in enumerate(boxes1):
            cell = image_segmented1[label_coords]
            if np.product(cell.shape) < 2000:  # Remove small labels
                image_segmented1 = np.where(label_img1 == label_ind + 1, 0, image_segmented1)

        # Regenerate labels for Image 1
        label_img1, nlabels1 = ndi.label(image_segmented1)
        st.write(f"After filtering, there are {nlabels1} separate components/objects detected in Image 1.")

        # Process Image 2
        boxes2 = ndi.find_objects(label_img2)
        for label_ind, label_coords in enumerate(boxes2):
            cell = image_segmented2[label_coords]
            if np.product(cell.shape) < 2000:  # Remove small labels
                image_segmented2 = np.where(label_img2 == label_ind + 1, 0, image_segmented2)

        # Regenerate labels for Image 2
        label_img2, nlabels2 = ndi.label(image_segmented2)
        st.write(f"After filtering, there are {nlabels2} separate components/objects detected in Image 2.")

        # Region properties for Image 1
        st.write("### Region Properties for Image 1")
        regions1 = regionprops(label_img1)

        fig1, ax1 = plt.subplots()
        ax1.imshow(image_segmented1, cmap=plt.cm.gray)
        for props in regions1:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

            ax1.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax1.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax1.plot(x0, y0, '.g', markersize=15)

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax1.plot(bx, by, '-b', linewidth=2.5)

        st.pyplot(fig1)

        # Region properties for Image 2
        st.write("### Region Properties for Image 2")
        regions2 = regionprops(label_img2)

        fig2, ax2 = plt.subplots()
        ax2.imshow(image_segmented2, cmap=plt.cm.gray)
        for props in regions2:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

            ax2.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax2.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax2.plot(x0, y0, '.g', markersize=15)

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax2.plot(bx, by, '-b', linewidth=2.5)

        st.pyplot(fig2)

    # Menghitung properti untuk label_img1
    props1 = regionprops_table(label_img1, properties=('centroid', 'orientation',
                                                   'major_axis_length', 'minor_axis_length'))
    df1 = pd.DataFrame(props1)

    # Menghitung properti untuk label_img2
    props2 = regionprops_table(label_img2, properties=('centroid', 'orientation',
                                                   'major_axis_length', 'minor_axis_length'))
    df2 = pd.DataFrame(props2)

    # Streamlit GUI
    st.title("Region Properties Display")

    # Menampilkan tabel untuk props1
    st.subheader("Properties for Image 1")
    st.dataframe(df1)

    # Menampilkan tabel untuk props2
    st.subheader("Properties for Image 2")
    st.dataframe(df2)
