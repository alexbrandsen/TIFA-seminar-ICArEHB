# by Ezequiel de la Rosa
# This script is adapted from
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "/Users/edelarosa/Documents/tifa/sam_vit_l_0b3195.pth" # download from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
model_type = "vit_l"
device = "cpu"


def onclick(event):
    global coordinates
    if event.button == 1:  # Left mouse button click
        # Store clicked coordinates
        coordinates.append([int(event.xdata), int(event.ydata)])
        print("Clicked at coordinates:", coordinates[-1])
        # Disconnect the mouse click event to close the window after one click
        plt.disconnect(cid)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



if __name__ == "__main__":
    ''' 
    Basic script to segment images using SAM.    
    '''

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    path_to_images = '/Users/edelarosa/Documents/tifa/axes/' #path to axes folder
    images = glob.glob(path_to_images + '/*.jpg')

    # create folder of labels

    if not os.path.exists(os.path.join(path_to_images, 'labels')):
        os.mkdir(os.path.join(path_to_images, 'labels'))

    # iterate over images
    for image_path in images:
        out_path = os.path.join(path_to_images, 'labels', image_path.split('/')[-1])
        if not os.path.exists(out_path):
            print('Running: ',image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a figure and plot the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('on')

            # Initialize empty list to store clicked coordinates
            coordinates = []

            # Connect the mouse click event to the function
            cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

            # Show the plot
            plt.show()

            # Convert clicked coordinates to numpy array
            input_point = np.array(coordinates)
            input_label = np.array([1])
            predictor.set_image(image)

            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            mask = masks[0, ...] #get mask

            # visualize segmentations
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(image)
            plt.subplot(1,2,2)
            plt.imshow(mask)
            plt.show()

            # Convert binary mask to uint8 (0 and 255) for visualization
            mask_uint8 = np.uint8(mask * 255)
            # Save the mask as a JPG image
            cv2.imwrite(out_path, mask_uint8)
            plt.imshow(mask)
