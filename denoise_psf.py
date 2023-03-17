import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_centroid(array_2d):
    if array_2d.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    # Create a meshgrid of coordinates
    y, x = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
    
    # Calculate the centroid
    total_intensity = array_2d.sum()
    centroid_x = (x * array_2d).sum() / total_intensity
    centroid_y = (y * array_2d).sum() / total_intensity
    
    return centroid_x, centroid_y
    
    
# Load dataset and preprocess
def load_data(path):
    files = glob.glob(os.path.join(path, '*.npy'))
    print(files)
    data = [np.load(f) for f in files]
    return np.array(data)

def add_noise(images, noise_factor=0.05):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0.0, 1.0)

def preprocess_data(path):
    images = load_data(path)
    images = images.astype(np.float32) * 20.0
    print(np.max(images))
    images_flat = images.reshape(-1, 120 * 120)
    noisy_images = add_noise(images_flat)
    return noisy_images, images_flat

# Create autoencoder model
def create_autoencoder(hidden_dim=64):
    input_img = Input(shape=(120 * 120,))
    encoded = Dense(hidden_dim, activation='relu')(input_img)
    decoded = Dense(120 * 120, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)
    return autoencoder
    
    
def create_autoencoder1(hidden_dim1=512, hidden_dim2=128):
    input_img = Input(shape=(120 * 120,))
    
    # Encoder
    encoded = Dense(hidden_dim1, activation='relu')(input_img)
    encoded = Dense(hidden_dim2, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(hidden_dim1, activation='relu')(encoded)
    decoded = Dense(120 * 120, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_img, decoded)
    return autoencoder
    

# Train the autoencoder
def train_autoencoder(path, hidden_dim=64, epochs=10, batch_size=32):
    # Load and preprocess data
    noisy_images, clean_images = preprocess_data(path)

    # Create the autoencoder model
    autoencoder = create_autoencoder(hidden_dim=hidden_dim)
    autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
    
    # Train the model
    autoencoder.fit(noisy_images, clean_images, epochs=epochs, batch_size=batch_size)

    return autoencoder
    
def visualize_results(clean_images, noisy_images, denoised_images, num_samples=5):
    clean_images = clean_images.reshape(-1, 120, 120)
    noisy_images = noisy_images.reshape(-1, 120, 120)
    denoised_images = denoised_images.reshape(-1, 120, 120)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))

    for i in range(num_samples):
        axes[i, 0].imshow(clean_images[i], cmap="gray")
        axes[i, 0].set_title("Clean Image")
        axes[i, 1].imshow(noisy_images[i], cmap="gray")
        axes[i, 1].set_title("Noisy Image")
        axes[i, 2].imshow(denoised_images[i], cmap="gray")
        axes[i, 2].set_title("Denoised Image")

    plt.show()

def visualize_results(clean_images, noisy_images, denoised_images, num_samples=8):
    clean_images = clean_images.reshape(-1, 120, 120)
    noisy_images = noisy_images.reshape(-1, 120, 120)
    denoised_images = denoised_images.reshape(-1, 120, 120)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))

    for i in range(num_samples):
        axes[i, 0].imshow(np.sqrt(clean_images[i]), cmap="gray")
        axes[i, 0].set_title("Clean Image")
        axes[i, 1].imshow(np.sqrt(noisy_images[i]), cmap="gray")
        axes[i, 1].set_title("Noisy Image")
        axes[i, 2].imshow(np.sqrt(denoised_images[i]), cmap="gray")
        axes[i, 2].set_title("Denoised Image")

    plt.show()
    
def calculate_metrics(clean_images, denoised_images):
    clean_images = clean_images.reshape(-1, 120, 120)
    denoised_images = denoised_images.reshape(-1, 120, 120)
    avg_psnr = np.mean([psnr(clean_images[i], denoised_images[i], data_range=1) for i in range(len(clean_images))])
    avg_ssim = np.mean([ssim(clean_images[i], denoised_images[i], data_range=1) for i in range(len(clean_images))])
    return avg_psnr, avg_ssim

def load_single_image(path):
    image = np.load(path)
    return image

def denoise_single_image(image, autoencoder, noise_factor=0.3):
    # Preprocess the image
    image = image.astype(np.float32) * 20.0
    noisy_image = add_noise(image.reshape(1, 120 * 120), noise_factor)
    
    # Denoise the image
    denoised_image_flat = autoencoder.predict(noisy_image)
    
    # Reshape the output back to its original dimensions
    denoised_image = denoised_image_flat.reshape(120, 120)
    
    return denoised_image

def test():
    single_image_path = "./psf/psfb8715.npy"
    single_image = load_single_image(single_image_path)

    # Denoise the single image
    denoised_single_image = denoise_single_image(single_image, autoencoder)

    # Visualize the result
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(single_image, cmap="gray")
    axes[0].set_title("Clean Image")
    axes[1].imshow(add_noise(single_image.reshape(120 * 120)).reshape(120, 120), cmap="gray")
    axes[1].set_title("Noisy Image")
    axes[2].imshow(denoised_single_image, cmap="gray")
    axes[2].set_title("Denoised Image")
    plt.show()
    
    
if __name__ == "__main__":
    # Set the path to your .npy files
    path_to_npy_files = "./psf/"
    #autoencoder.load("autoencoder.h5")
    # Train the autoencoder
    autoencoder = train_autoencoder(path_to_npy_files, hidden_dim=128, epochs=230, batch_size=128)

    # Save the trained model
    autoencoder.save("autoencoder.h5")

    # Test the autoencoder
    test_noisy_images, test_clean_images = preprocess_data("./psf/")
    denoised_images = autoencoder.predict(test_noisy_images)

    # Visualize the results
    visualize_results(test_clean_images, test_noisy_images, denoised_images)

    # Calculate PSNR and SSIM
    avg_psnr, avg_ssim = calculate_metrics(test_clean_images, denoised_images)
    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
