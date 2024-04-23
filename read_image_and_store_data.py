import numpy as np
import pandas as pd
import glob
import imageio

from skimage import measure
import matplotlib.pyplot as plt

Lx, Ly = 0.1e-6, 0.1e-6 #0.1 micrometers

def get_variant_files():
    variant_files = []
    for i in range(1, 13):
        variant_string = f"A01_XY_100_nm/variant_{i}/A01_XY_100_nm_vId_{i}.png"
        variant_files.append(variant_string)
    return variant_files

def read_images(FILES, crop_top, crop_bottom, crop_left, crop_right):
    images = []
    for FILE in FILES:
        image = imageio.imread(FILE)
        cropped_image = image[crop_top:-crop_bottom, crop_left:-crop_right]
        images.append(cropped_image)
    return images

def filter_image(image, filter_size):
    labeled_image = measure.label(image)
    blobs = measure.regionprops(labeled_image)
    blobs_sizes = np.array([blob.num_pixels for blob in blobs])
    remove_idxs = np.argwhere(blobs_sizes<filter_size).flatten()
    filtered_blobs = [blobs[i] for i in remove_idxs]
    for filtered_blob in filtered_blobs:
        image[filtered_blob.coords[:,0], filtered_blob.coords[:,1]] = 0.

    labeled_image = measure.label(image)
    blobs = measure.regionprops(labeled_image)
    return image, labeled_image, blobs

def get_lath_data(image, alpha_variant, filter_size=80, display_labeled_image=False):
    """
    This function recoveres:
        1. lath sizes (equivalent diameters), 
        2. coordinates (assume (x,y), pixel) of the centroid
        3. axis major and minor lengths (pixels)
        4. orientation (rotation about out of plane, radians)
    """
    image = image.mean(axis=2) #makes black and white image
    image, labeled_image, blobs = filter_image(image, filter_size)
    if display_labeled_image:
        plot_labeled_image(labeled_image, alpha_variant)
    image_height, image_width = labeled_image.shape
    x_scale = Lx / image_width
    y_scale = Ly / image_height
    areas = np.empty(len(blobs))
    equivalent_diameters = np.empty(len(blobs))
    coordinates = np.empty((len(blobs),2))
    major_axis_lengths = np.empty(len(blobs))
    minor_axis_lengths = np.empty(len(blobs))
    orientations = np.empty(len(blobs))
    #for i, blob in enumerate(blobs):
        #blob.area *= x_scale * y_scale
        #blob.centroid = tuple(coord * np.array([y_scale, x_scale]) for coord in blob.centroid)
        #blob.bbox = tuple(coord * np.array([y_scale, x_scale]) for coord in blob.bbox)
        #areas[i] = blob.area * x_scale * y_scale
        #equivalent_diameters[i] = np.sqrt(areas[i]/np.pi) / 2.0
        #equivalent_diameters[i] = blob.equivalent_diameter
        #coordinates[i] = blob.centroid * np.array([y_scale, x_scale])
        #major_axis_lengths[i] = blob.major_axis_length * max((x_scale, y_scale))
        #minor_axis_lengths[i] = blob.minor_axis_length * min((x_scale, y_scale))
        #orientations[i] = blob.orientation

    areas = np.array([blob.area for blob in blobs])
    equivalent_diameters = np.array([blob.equivalent_diameter for blob in blobs])
    coordinates = np.array([blob.centroid for blob in blobs])
    major_axis_lengths = np.array([blob.axis_major_length for blob in blobs])
    minor_axis_lengths = np.array([blob.axis_minor_length for blob in blobs])
    orientations = np.array([blob.orientation for blob in blobs])
    alpha_variant = (np.ones(orientations.shape) * alpha_variant).astype(int)
    information = {"areas":areas,
                   "equivalent_diameter":equivalent_diameters,
                   "centroid_x":coordinates[:,0],
                   "centroid_y":coordinates[:,1],
                   "major_axis_length":major_axis_lengths,
                   "minor_axis_length":minor_axis_lengths,
                   "orientation":orientations,
                   "alpha_variant":alpha_variant}
    df = pd.DataFrame.from_dict(information)
    return df

def plot_labeled_image(labeled_image, variant):
    plt.figure(figsize=(8, 6))
    plt.imshow(labeled_image)
    plt.colorbar(label='Lath Id')
    plt.title(f"Alpha Variant: {variant+1}")
    plt.show()
    
if __name__ == "__main__":
    
    
    variant_files = get_variant_files()
    images = read_images(variant_files, 
                         crop_top=26, 
                         crop_bottom=140,
                         crop_left=182, 
                         crop_right=182)
    lath_data = [get_lath_data(image, 
                               alpha_variant=i+1,
                                filter_size=80, 
                                display_labeled_image=False) \
                                for i, image in enumerate(images)]

    information = pd.concat(lath_data, ignore_index=True)

    information.to_csv('single_scan.csv', index=False)
    import pdb;pdb.set_trace()
