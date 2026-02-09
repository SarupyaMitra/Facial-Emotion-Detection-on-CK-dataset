import numpy as np
from Gabor_Outputs_generation import Gabor_filter_data,csv_name

# In this file, we will use the previously saved outputs and extract features.
# We are going to be using blockwise statistics from each of the outputs of each input image.
# We then save the features.

def split_into_blocks(op_img,block_shape):
    no_of_ht_pixels_in_block = op_img.shape[0]//block_shape[0]
    no_of_wd_pixels_in_block = op_img.shape[1]//block_shape[1]

    blocks = []
    # We are looking for non-overlapping blocks
    for i in range(0,op_img.shape[0],no_of_ht_pixels_in_block):
        for j in range(0,op_img.shape[1],no_of_wd_pixels_in_block):
            block = op_img[i:i+no_of_ht_pixels_in_block , j: j + no_of_wd_pixels_in_block]
            blocks.append(block)
    
    return blocks

def extract_features(blocks):
    means = [] 
    for block in blocks:
        mean = block.mean()
        means.append(mean)

    return means

def find_classification_ready_features(outputs):
    # Break the outputs in 4*4 grid. Every grid must give a statistic (mean of the orientation energy)
    imgwise_blockmeans = []
    #blockwise_var = []
    # Every output is 38*38, now to break the output into 4*4=16 grids, we need to change the output spatial dimensions to a no. which is multiple of 4
    # So we select the central 36*36 pixels from 38*38 size. We will disregard the topmost,bottomest,leftmost,rightmost pixels
    # So we can break 36*36 output to 4*4 blocks with each block having 9*9 pixels.
    grid_shape = (4,4) 
    for img in outputs: 
        blockwise_means = []
        for op in img:
            op = op[1:-1,1:-1] # Disregrading topmost,bottomest,leftmost,rightmost pixels
            blocks = split_into_blocks(op,grid_shape)
            # There are total 16 blocks per output image. Hence each output image will give [16*(no. of statistic)]  features.
            means = extract_features(blocks)
            blockwise_means.extend(means)     # 16 blocks in an op img. So 16 means stored together. extend is used because it adds contents of a list to another.
            # That way we get 16(no_of_Gabor_outputs) * 16 (blocks_per_op_img) = 256 = len(blockwise_means)

        imgwise_blockmeans.append(blockwise_means)  # We then append this 256-length  blockwise-means here. So at the end we will get 
        # imgwise_blockmeans.shape = no_of_ip_img * (no_of_Gabor_outputs * blocks_per_op_img) = no_of_ip_img * no_of_features_per_ip_img


    imgwise_blockmeans = np.array(imgwise_blockmeans)
    print(f"Interest ---> {imgwise_blockmeans.shape}")

    #blockwise_means = blockwise_means.reshape((920,no_of_features_per_input_img))  # Bcoz for each i/p img we have 256 blockwise mean values.
    return imgwise_blockmeans


def main():
    # Extracting the saved outputs
    if csv_name == "data\\fer2013.csv":
        outputs = np.load("data\\outputs_fer.npy")
    elif csv_name == "data\\ckextended.csv":
        outputs = np.load("data\\outputs_ck.npy")
    print(f"The outputs shape is = {outputs.shape}.(No. of input images * no. of Filters/outputs * output ht * output wd)")
    imgwise_blockmeans = find_classification_ready_features(outputs)
    if csv_name == "data\\fer2013.csv":
        np.save("data\\imgwise_blockmeans_fer.npy",imgwise_blockmeans)
    elif csv_name == "data\\ckextended.csv":
        np.save("data\\imgwise_blockmeans_ck.npy",imgwise_blockmeans)
    print("Image wise Block Statistics Successfully Saved")                                                                                                                                                                                                               

if __name__ == '__main__':
    main()

























