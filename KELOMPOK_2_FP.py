import streamlit as st
import imageio
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
from PIL import Image

image_segmented1 = None
image_segmented2 = None

# Fungsi untuk mengonversi gambar menjadi grayscale
def convert_to_grayscale(image):
    # Menggunakan AHE untuk meningkatkan kontras
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.01)
    # Menghitung grayscale dari gambar yang sudah di AHE
    my_gray = img_adapteq @ np.array([0.2126, 0.7152, 0.0722])
    my_gray = img_as_ubyte(my_gray)
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
menu = st.sidebar.radio("Menu", ["Home", "Image Details", "Histograms", "AHE Results", "Grayscale & Median Filter", "Thresholding", "Penghitungan dan Visualisasi Histogram Gambar yang Difilter", "region"])

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
    col1, col2 = st.columns(2)
    
    if im1 is not None:
        im1_array = np.array(im1)  # Konversi gambar 1 ke array NumPy
        with col1:
            st.write("### Details for Image 1:")
            st.write(f"Type: {type(im1)}")
            st.write(f"dtype: {im1_array.dtype}")  # dtype dari array NumPy
            st.write(f"shape: {im1_array.shape}")  # shape dari array NumPy
            st.write(f"size: {im1.size}")  # size tetap dari PIL Image (lebar, tinggi)
            st.image(im1, caption="Image 1", use_column_width=True)
    else:
        with col1:
            st.write("No Image 1 uploaded.")
    
    if im2 is not None:
        im2_array = np.array(im2)  # Konversi gambar 2 ke array NumPy
        with col2:
            st.write("### Details for Image 2:")
            st.write(f"Type: {type(im2)}")
            st.write(f"dtype: {im2_array.dtype}")  # dtype dari array NumPy
            st.write(f"shape: {im2_array.shape}")  # shape dari array NumPy
            st.write(f"size: {im2.size}")  # size tetap dari PIL Image (lebar, tinggi)
            st.image(im2, caption="Image 2", use_column_width=True)
    else:
        with col2:
            st.write("No Image 2 uploaded.")

# Histogram Gambar
elif menu == "Histograms":
    st.write("## Histograms for Images")
    col1, col2 = st.columns(2)
    
    if im1 is not None:
        im1_array = np.array(im1)  # Konversi gambar ke array NumPy
        with col1:
            st.write("### Histogram for Image 1")
            plot_histogram(im1_array, 'Histogram for Image 1', (0, 255), (0, 15000))
    else:
        with col1:
            st.write("No Image 1 uploaded.")
    
    if im2 is not None:
        im2_array = np.array(im2)  # Konversi gambar ke array NumPy
        with col2:
            st.write("### Histogram for Image 2")
            plot_histogram(im2_array, 'Histogram for Image 2', (0, 255), (0, 15000))
    else:
        with col2:
            st.write("No Image 2 uploaded.")

# Bagian Hasil AHE
elif menu == "AHE Results":
    st.write("## Adaptive Histogram Equalization (AHE) Results")
    if im1 is not None:
        img_adapteq1 = ahe_processing(np.array(im1))  # AHE pada array gambar
    else:
        img_adapteq1 = None
        
    if im2 is not None:
        img_adapteq2 = ahe_processing(np.array(im2))  # AHE pada array gambar
    else:
        img_adapteq2 = None

    col1, col2 = st.columns(2)
    
    # Tampilkan hasil AHE dan gambar asli untuk Image 1
    with col1:
        if im1 is not None:
            st.write("### Original Image 1")
            st.image(im1, caption='Original Image 1', use_column_width=True)
            st.write("### AHE Image 1")
            st.image(img_adapteq1.astype(np.uint8), caption='AHE Image 1', use_column_width=True)
        else:
            st.write("No Image 1 uploaded.")
    
    # Tampilkan hasil AHE dan gambar asli untuk Image 2
    with col2:
        if im2 is not None:
            st.write("### Original Image 2")
            st.image(im2, caption='Original Image 2', use_column_width=True)
            st.write("### AHE Image 2")
            st.image(img_adapteq2.astype(np.uint8), caption='AHE Image 2', use_column_width=True)
        else:
            st.write("No Image 2 uploaded.")

    if im1 is not None and img_adapteq1 is not None and im2 is not None and img_adapteq2 is not None:
        st.write("### Histograms Comparison for AHE")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Plot original and AHE histograms for Image 1
        axes[0, 0].plot(ndi.histogram(np.array(im1), min=0, max=255, bins=256), color='orange')
        axes[0, 0].set_title("Histogram for Image 1")
        axes[0, 1].plot(ndi.histogram(img_adapteq1, min=0, max=255, bins=256), color='blue')
        axes[0, 1].set_title("Histogram for AHE Image 1")

        # Plot original and AHE histograms for Image 2
        axes[1, 0].plot(ndi.histogram(np.array(im2), min=0, max=255, bins=256), color='orange')
        axes[1, 0].set_title("Histogram for Image 2")
        axes[1, 1].plot(ndi.histogram(img_adapteq2, min=0, max=255, bins=256), color='blue')
        axes[1, 1].set_title("Histogram for AHE Image 2")

        st.pyplot(fig)

# Bagian Grayscale dan Median Filter
elif menu == "Grayscale & Median Filter":
    st.write("## Grayscale & Median Filter")

    if im1 is not None:
        my_gray1 = convert_to_grayscale(np.array(im1))  # Konversi gambar 1 ke grayscale
        med1 = MedianFilter(my_gray1)  # Median Filter pada grayscale Image 1
    else:
        my_gray1 = med1 = None
    
    if im2 is not None:
        my_gray2 = convert_to_grayscale(np.array(im2))  # Konversi gambar 2 ke grayscale
        med2 = MedianFilter(my_gray2)  # Median Filter pada grayscale Image 2
    else:
        my_gray2 = med2 = None
    
    col1, col2 = st.columns(2)
    
    # Tampilkan grayscale dan hasil median filter untuk Image 1
    with col1:
        if my_gray1 is not None:
            st.write("### Grayscale Image 1")
            st.image(my_gray1.astype(np.uint8), caption="Grayscale Image 1", use_column_width=True)
            st.write("### Median Filtered Image 1")
            st.image(med1.astype(np.uint8), caption="Median Filtered Image 1", use_column_width=True)
        else:
            st.write("No Image 1 uploaded.")
    
    # Tampilkan grayscale dan hasil median filter untuk Image 2
    with col2:
        if my_gray2 is not None:
            st.write("### Grayscale Image 2")
            st.image(my_gray2.astype(np.uint8), caption="Grayscale Image 2", use_column_width=True)
            st.write("### Median Filtered Image 2")
            st.image(med2.astype(np.uint8), caption="Median Filtered Image 2", use_column_width=True)
        else:
            st.write("No Image 2 uploaded.")
    
    # Plot histogram untuk gambar yang difilter
    if med1 is not None and med2 is not None:
        histogram_med1 = ndi.histogram(med1, min=0, max=255, bins=256)
        histogram_med2 = ndi.histogram(med2, min=0, max=255, bins=256)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Histogram Median Filtered Image 1")
            fig1, ax1 = plt.subplots()
            ax1.plot(histogram_med1)
            st.pyplot(fig1)
        with col2:
            st.write("### Histogram Median Filtered Image 2")
            fig2, ax2 = plt.subplots()
            ax2.plot(histogram_med2)
            st.pyplot(fig2)
    
# Bagian Thresholding 
elif menu == "Thresholding":
    st.write("## Thresholding")
    
    # Memastikan gambar grayscale dan median filter sudah ada
    if im1 is not None and im2 is not None:
        # Mengonversi ke grayscale dan menerapkan Median Filter jika belum ada
        if 'med1' not in locals() or med1 is None:
            my_gray1 = convert_to_grayscale(np.array(im1))
            med1 = MedianFilter(my_gray1)
        
        if 'med2' not in locals() or med2 is None:
            my_gray2 = convert_to_grayscale(np.array(im2))
            med2 = MedianFilter(my_gray2)
    
        # Menampilkan properti gambar
        st.write("### Image Properties:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Image 1 Properties:")
            st.write(f"Type: {type(med1)}")
            st.write(f"dtype: {med1.dtype}")
            st.write(f"shape: {med1.shape}")
            st.write(f"size: {med1.size}")
        
        with col2:
            st.write("#### Image 2 Properties:")
            st.write(f"Type: {type(med2)}")
            st.write(f"dtype: {med2.dtype}")
            st.write(f"shape: {med2.shape}")
            st.write(f"size: {med2.size}")

        # Thresholding menggunakan Otsu
        from skimage import filters
        
        threshold1 = filters.threshold_otsu(med1)
        threshold2 = filters.threshold_otsu(med2)

        # Menampilkan nilai threshold Otsu
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Image 1:")
            st.write("Nilai threshold Otsu untuk gambar 1:", threshold1)
        
        with col2:
            st.write("#### Image 2:")
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
    
    else:
        st.write("Please upload both images to apply thresholding and median filtering.")

# Bagian Penghitungan dan Visualisasi Histogram Gambar yang Difilter
elif menu == "Penghitungan dan Visualisasi Histogram Gambar yang Difilter":
    st.write("## Edge Detection of Median Filtered Images")

    # Memastikan gambar yang terfilter sudah ada
    if im1 is not None and im2 is not None:
        if 'med1' not in locals() or med1 is None:
            my_gray1 = convert_to_grayscale(np.array(im1))
            med1 = MedianFilter(my_gray1)

        if 'med2' not in locals() or med2 is None:
            my_gray2 = convert_to_grayscale(np.array(im2))
            med2 = MedianFilter(my_gray2)

        # Slice pada median_filtered11 (cropped area from med1)
        median_filtered11 = med1[:, 100:]

        # Normalisasi dan tampilkan gambar median_filtered11 di Streamlit
        st.image(normalize_image(median_filtered11), caption="Median Filtered 11 (Cropped)", use_column_width=True)

        # Thresholding dan contour untuk median_filtered11
        threshold3 = filters.threshold_otsu(median_filtered11)
        fig, ax = plt.subplots()
        ax.imshow(median_filtered11, cmap='gray')
        ax.contour(median_filtered11, [threshold3], colors='red')
        ax.set_title(f'Contour for Median Filtered 11 at Threshold {threshold3}')
        st.pyplot(fig)

        # Thresholding dan contour untuk median_filtered2 (full med2 image)
        median_filtered2 = med2
        threshold4 = filters.threshold_otsu(median_filtered2)
        fig, ax = plt.subplots()
        ax.imshow(median_filtered2, cmap='gray')
        ax.contour(median_filtered2, [threshold4], colors='red')
        ax.set_title(f'Contour for Median Filtered 2 at Threshold {threshold4}')
        st.pyplot(fig)

        # Binary classification
        st.write("## Binary Classification")
        st.header('Image 1')
        # Layout tiga kolom
        col1, col2, col3 = st.columns(3)
        # Binary Classification Image 1 di kolom pertama
        with col1:
            st.subheader('Binary Image')
            binary_image1 = median_filtered11 < threshold3
            st.image(binary_image1.astype(np.uint8) * 255, caption="Binary Classification Image 1", use_column_width=True)

        # Large Blobs Image 1 di kolom kedua
        with col2:
            st.subheader('Large Blobs')
            only_large_blobs1 = morphology.remove_small_objects(binary_image1, min_size=100)
            st.image(only_large_blobs1.astype(np.uint8) * 255, caption="Large Blobs Image 1", use_column_width=True)

        # Segmented Image 1 di kolom ketiga
        with col3:
            st.subheader('Segmented')
            only_large1 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs1), min_size=100))
            image_segmented1 = only_large1
            st.image(image_segmented1.astype(np.uint8) * 255, caption="Segmented Image 1", use_column_width=True)
       


        #IMAGE 2
        # Layout tiga kolom untuk IMAGE 2
        st.header('Image 2')
        col1, col2, col3 = st.columns(3)

        # Binary Classification Image 2 di kolom pertama
        with col1:
            st.subheader('Binary Image')
            binary_image2 = median_filtered2 < threshold4
            st.image(binary_image2.astype(np.uint8) * 255, caption="Binary Classification Image 2", use_column_width=True)

        #   Large Blobs Image 2 di kolom kedua
        with col2:
            st.subheader('Large Blobs')
            only_large_blobs2 = morphology.remove_small_objects(binary_image2, min_size=100)
            st.image(only_large_blobs2.astype(np.uint8) * 255, caption="Large Blobs Image 2", use_column_width=True)

        # Segmented Image 2 di kolom ketiga
        with col3:
            st.subheader('Segmented')
            only_large2 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs2), min_size=100))
            image_segmented2 = only_large2
            st.image(image_segmented2.astype(np.uint8) * 255, caption="Segmented Image 2", use_column_width=True)
        

        # Calculate histograms for both filtered images
        st.write("## Histogram")
        histo_median1 = ndi.histogram(med1, min=0, max=255, bins=256)
        histo_median2 = ndi.histogram(med2, min=0, max=255, bins=256)

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

    else:
        st.write("Please upload both images to calculate and visualize histograms.")

# region
elif menu == "region":
    st.write("## Region Analysis of Segmented Images")

    # Jika image_segmented1 atau image_segmented2 belum didefinisikan, lakukan proses segmentasi
    if image_segmented1 is None or image_segmented2 is None:
        # Pastikan median_filtered1 dan median_filtered2 sudah ada
        if 'med1' not in locals() or 'med2' not in locals():
            # Lakukan median filtering jika belum dilakukan
            my_gray1 = convert_to_grayscale(np.array(im1))
            my_gray2 = convert_to_grayscale(np.array(im2))
            med1 = MedianFilter(my_gray1)
            med2 = MedianFilter(my_gray2)
    
        # Lakukan segmentasi pada gambar yang sudah difilter
        only_large_blobs1 = morphology.remove_small_objects(med1 < filters.threshold_otsu(med1), min_size=100)
        only_large_blobs2 = morphology.remove_small_objects(med2 < filters.threshold_otsu(med2), min_size=100)
        
        image_segmented1 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs1), min_size=100))
        image_segmented2 = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs2), min_size=100))
    
    # Label the segmented images
    label_img1, nlabels1 = ndi.label(image_segmented1)
    label_img2, nlabels2 = ndi.label(image_segmented2)

    # Tampilkan informasi mengenai jumlah komponen yang terdeteksi
    st.write(f"For Image 1, there are {nlabels1} separate components/objects detected.")
    st.write(f"For Image 2, there are {nlabels2} separate components/objects detected.")

    # Proses Image 1 untuk menghapus label kecil
    boxes1 = ndi.find_objects(label_img1)
    for label_ind, label_coords in enumerate(boxes1):
        cell = image_segmented1[label_coords]
        if np.product(cell.shape) < 2000:  # Filter komponen yang terlalu kecil
            image_segmented1 = np.where(label_img1 == label_ind + 1, 0, image_segmented1)
    
    # Regenerasi label untuk Image 1 setelah filtering
    label_img1, nlabels1 = ndi.label(image_segmented1)
    st.write(f"After filtering, there are {nlabels1} separate components/objects detected in Image 1.")
    
    # Proses Image 2 untuk menghapus label kecil
    boxes2 = ndi.find_objects(label_img2)
    for label_ind, label_coords in enumerate(boxes2):
        cell = image_segmented2[label_coords]
        if np.product(cell.shape) < 2000:  # Filter komponen yang terlalu kecil
            image_segmented2 = np.where(label_img2 == label_ind + 1, 0, image_segmented2)
    
    # Regenerasi label untuk Image 2 setelah filtering
    label_img2, nlabels2 = ndi.label(image_segmented2)
    st.write(f"After filtering, there are {nlabels2} separate components/objects detected in Image 2.")
    
    # Region properties untuk Image 1
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

        # Plot centroid and orientation lines
        ax1.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax1.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax1.plot(x0, y0, '.g', markersize=15)

        # Plot bounding box
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax1.plot(bx, by, '-b', linewidth=2.5)

    # Set limit sumbu x dan y
    ax1.set_xlim(0, 700)
    ax1.set_ylim(480, 0)  # Note: for images, y-axis is usually inverted
    st.pyplot(fig1)
    

    # Region properties untuk Image 2
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

        # Plot centroid and orientation lines
        ax2.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax2.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax2.plot(x0, y0, '.g', markersize=15)

        # Plot bounding box
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax2.plot(bx, by, '-b', linewidth=2.5)
    
    # Set limit sumbu x dan y
    ax2.set_xlim(0, 700)
    ax2.set_ylim(470, 0)  # y-axis inverted
    st.pyplot(fig2)

    # Menghitung properti untuk label_img1
    props1 = regionprops_table(label_img1, properties=('centroid', 'orientation',
                                                       'major_axis_length', 'minor_axis_length'))
    df1 = pd.DataFrame(props1)

    # Menghitung properti untuk label_img2
    props2 = regionprops_table(label_img2, properties=('centroid', 'orientation',
                                                       'major_axis_length', 'minor_axis_length'))
    df2 = pd.DataFrame(props2)

    # Streamlit GUI untuk menampilkan hasil properti
    st.title("Region Properties Display")

    # Tampilkan tabel untuk properti Image 1
    st.subheader("Properties for Image 1")
    st.dataframe(df1)

    # Tampilkan tabel untuk properti Image 2
    st.subheader("Properties for Image 2")
    st.dataframe(df2)