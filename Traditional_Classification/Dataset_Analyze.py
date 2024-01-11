#import the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image

def read_img_data(path, img_width, img_height):
    filepaths = []
    labels = []
    images = []
    sizes = []

    # Load all the filenames and labels
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        filepath = os.path.join(path, file)
        try:
            img = Image.open(filepath)
            # retrive the size of the image
            original_size = img.size
            # Convert the image to RGB if it's not already in that format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((img_width, img_height))
            img_array = np.array(img)
            # Ensure all images have the same shape (256, 256, 3)
            if img_array.shape == (img_height, img_width, 3):
                label = file.split('_')[0]
                images.append(img_array)
                labels.append(label)
                filepaths.append(filepath)
                sizes.append((label, original_size))
            else:
                print(f"Image {file} has an inconsistent shape: {img_array.shape}")
        except Exception as e:
            print(f"Error loading image {file}: {e}")

    # Convert the image data to a NumPy array, set the unique labels
    unique_labels = set(labels)
    images = np.array(images)

    # Plot the distribution of labels in the dataset
    label_counts={label: 0 for label in unique_labels}
    print(unique_labels)

    fixed_colors = np.array([[1.0, 0.0, 0.0],  # red
                         [1.0, 0.5, 0.0],  # orange
                         [1.0, 1.0, 0.0],  # yellow
                         [0.0, 1.0, 0.0]])  # green

    random_colors = np.random.rand(len(label_counts) - 4, 3)

    colors = np.vstack((fixed_colors, random_colors))

    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    labels_unique = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels_unique, counts, color=colors)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Labels in Dataset')
    plt.show()

    #plot the distribution of image sizes
    size_dict = {label: [] for label in unique_labels}
    for label, size in sizes:
        size_dict[label].append(size)
    
    avg_sizes = {}
    for label in unique_labels:
        widths, heights = zip(*size_dict[label])
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        avg_sizes[label] = (avg_width, avg_height)

    plt.figure(figsize=(8, 5))
    for i, label in enumerate(unique_labels):
        label_sizes = size_dict[label]
        if label_sizes:
            x, y = zip(*label_sizes)
            plt.scatter(x, y, label=label)

            avg_x, avg_y = avg_sizes[label]
            plt.scatter(avg_x, avg_y, color='black', marker='x', s=100)
    
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Size Distribution of Images in Each Category with Average Points')
    plt.legend()
    plt.show()

    widths = {label: [size[0] for size in sizes] for label, sizes in size_dict.items()}
    heights = {label: [size[1] for size in sizes] for label, sizes in size_dict.items()}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for i, label in enumerate(unique_labels):
        axes[0].boxplot(widths[label], positions=[i+1], widths=0.6, patch_artist=True)
    axes[0].set_xticks(range(1, len(unique_labels) + 1))
    axes[0].set_xticklabels(unique_labels)
    axes[0].set_title('Widths of Each Category')
    axes[0].set_ylabel('Width (pixels)')

    for i, label in enumerate(unique_labels):
        axes[1].boxplot(heights[label], positions=[i+1], widths=0.6, patch_artist=True)
    axes[1].set_xticks(range(1, len(unique_labels) + 1))
    axes[1].set_xticklabels(unique_labels)
    axes[1].set_title('Heights of Each Category')
    axes[1].set_ylabel('Height (pixels)')

    plt.tight_layout()
    plt.show()
    
    return images, labels, unique_labels

def analyze_dataset_color(images, labels, unique_labels):
    label_colors = {}
    
    num_labels = len(unique_labels)
    num_cols = int(np.sqrt(num_labels))
    num_rows = num_labels // num_cols + (num_labels % num_cols > 0)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    if num_labels == 1:
        axs = [axs]
    axs = np.array(axs).reshape(num_rows, num_cols)

    for idx, label in enumerate(unique_labels):
        label_images = images[np.where(np.array(labels) == label)]
        avg_color = np.mean(label_images, axis=(0, 1, 2))
        label_colors[label] = avg_color

        row, col = idx // num_cols, idx % num_cols
        ax = axs[row, col]

        ax.bar(['Red', 'Green', 'Blue'], avg_color, color=['r', 'g', 'b'])
        ax.set_title(f'{label} Average Color')
        ax.set_ylabel('Intensity')

    plt.tight_layout()
    plt.show()

from matplotlib import cm
import numpy as np

def analyze_dataset_color_histogram(images, labels, unique_labels):
    histograms = {label: {channel: [] for channel in ['r', 'g', 'b']} for label in unique_labels}

    for img, label in zip(images, labels):
        for channel in range(3):
            channel_hist = np.histogram(img[:, :, channel], bins=256, range=(0, 255))[0]
            histograms[label]['rgb'[channel]].append(channel_hist)

    avg_histograms = {label: {channel: np.mean(histograms[label][channel], axis=0) for channel in ['r', 'g', 'b']} for label in unique_labels}

    for label in unique_labels:
        plt.figure(figsize=(10, 5))
        for channel, color in zip(['r', 'g', 'b'], ['red', 'green', 'blue']):
            plt.plot(avg_histograms[label][channel], color=color, label=f'{channel.upper()} channel')
            peak = np.argmax(avg_histograms[label][channel])
            plt.plot(peak, avg_histograms[label][channel][peak], 'o', color=color)

        plt.title(f'Average RGB Histogram for {label}')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def analyze_dataset_color_3D(images, labels, unique_labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    color_map = cm.get_cmap('rainbow', len(unique_labels))
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

    for img, label in zip(images, labels):
        color = label_to_color[label]
        ax.scatter(img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(), color=color)

    for label, color in label_to_color.items():
        ax.scatter([], [], [], color=color, label=label)

    ax.set_xlabel('Red Intensity')
    ax.set_ylabel('Green Intensity')
    ax.set_zlabel('Blue Intensity')
    ax.set_title('Image Color Intensity by Label')
    ax.legend()

    plt.show()

def prep_data(images, labels, unique_labels):
    x = images
    y = []

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    for label in labels:
        label_index = label_to_index[label]
        y_onehot = encode_onehot(label_index, len(unique_labels))
        y.append(y_onehot)

    return np.array(x), np.array(y)

def encode_onehot(label_idx, num_classes):
    onehot_encoded = np.zeros(num_classes)
    onehot_encoded[label_idx] = 1
    return onehot_encoded

def analyze_img_histogram():
    img = Image.open('CA/Traditional_Classification/test/orange_81.jpg')
    img = img.convert('RGB')
    img_array = np.array(img)

    # Create a mask for non-white pixels
    non_white_mask = ~(np.all(img_array == [255, 255, 255], axis=-1))

    # Apply the mask to filter out the white pixels
    filtered_img_array = img_array[non_white_mask]

    # Save the filtered array to a text file, if needed
    #np.savetxt('my_non_white_image_array.txt', filtered_img_array.reshape(-1, 3), fmt='%d', header='RGB values')

    # Calculate the histograms for each channel using the filtered array
    total_non_white_pixels = filtered_img_array.shape[0]
    proportion_hist_r = np.histogram(filtered_img_array[:, 0], bins=256, range=(0, 255))[0] / total_non_white_pixels
    proportion_hist_g = np.histogram(filtered_img_array[:, 1], bins=256, range=(0, 255))[0] / total_non_white_pixels
    proportion_hist_b = np.histogram(filtered_img_array[:, 2], bins=256, range=(0, 255))[0] / total_non_white_pixels

    # Plotting the normalized histograms
    plt.plot(proportion_hist_r, color='red', label='Red Channel')
    plt.plot(proportion_hist_g, color='green', label='Green Channel')
    plt.plot(proportion_hist_b, color='blue', label='Blue Channel')

    plt.title('Normalized Histogram of RGB Channels (Excluding White Pixels)')
    plt.xlabel('Intensity Value')
    plt.ylabel('Proportion of Non-White Pixels')
    plt.legend()
    plt.show()

def analyze_dataset_color_histogram_non_white(images, labels, unique_labels):
    # Initialize histograms structure
    histograms = {label: {channel: [] for channel in ['r', 'g', 'b']} for label in unique_labels}

    # Compute histograms for each category
    for img, label in zip(images, labels):
        # Create a mask for non-white pixels
        non_white_mask = ~(np.all(img == [255, 255, 255], axis=-1))

        for channel in range(3):
            # Apply the mask to filter out the white pixels for each channel
            filtered_img_channel = img[..., channel][non_white_mask]
            # Compute the histogram for non-white pixels of this channel
            channel_hist = np.histogram(filtered_img_channel, bins=256, range=(0, 255))[0]
            histograms[label]['rgb'[channel]].append(channel_hist)

    # Calculate average histograms
    avg_histograms = {label: {channel: np.mean(histograms[label][channel], axis=0)
                              for channel in ['r', 'g', 'b']}
                      for label in unique_labels}

    # Determine the layout for subplots based on the number of unique labels
    num_unique_labels = len(unique_labels)
    
    # Create subplots - one subplot for each unique label
    fig, axes = plt.subplots(nrows=num_unique_labels, ncols=1, figsize=(6, 3 * num_unique_labels), constrained_layout=True)

    # If there's only one label, wrap the axes in a list for consistent indexing
    if num_unique_labels == 1:
        axes = [axes]

    # Plot average histograms for each category in separate subplots
    for idx, label in enumerate(unique_labels):
        for channel, color in zip(['r', 'g', 'b'], ['red', 'green', 'blue']):
            # Normalizing the histogram by the number of images to get the average pixel count
            axes[idx].plot(avg_histograms[label][channel], color=color, label=f'{channel.upper()} channel')
            # Mark the peak of the average histogram
            peak = np.argmax(avg_histograms[label][channel])
            axes[idx].plot(peak, avg_histograms[label][channel][peak], 'o', color=color)
            axes[idx].set_title(f'Average RGB Histogram for {label}')
            axes[idx].set_xlabel('Intensity Value')
            axes[idx].set_ylabel('Average Pixel Count')
            axes[idx].legend()

    # Show the plot
    plt.show()

#please alter this path to your own path
train_data_dir = 'CA/Traditional_Classification/train'
test_data_dir = 'CA/Traditional_Classification/test'
img_width = 256
img_height = 256

def main():
    train_images, labels, unique_labels = read_img_data(train_data_dir, img_width, img_height)
    #analyze_dataset_color_histogram_non_white(train_images, labels, unique_labels)
    #analyze_dataset_color(train_images, labels, unique_labels)
    #analyze_dataset_color_histogram(train_images, labels, unique_labels)
    analyze_dataset_color_3D(train_images, labels, unique_labels)
    #analyze_img_histogram()

if __name__ == "__main__":
    main()