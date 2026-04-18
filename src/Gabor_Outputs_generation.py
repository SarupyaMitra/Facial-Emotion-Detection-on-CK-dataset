import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Here we are just fetching the input images, applying Gabor filter to each input images
# Then saving the outputs.

csv_name = "src\\data\\ckextended.csv"

def filter_def(size,lamb,gamma,sigma,psi,theta): 
# lamb -> wavelength; gamma -> ellipticity ; sigma -> gaussian std deviation ; psi -> phase ; theta -> angle
    center = size//2
    kernel = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            x_rotated = x*np.cos(theta) + y*np.sin(theta)
            y_rotated = -x*np.sin(theta) + y*np.cos(theta)
            gaussian = (math.exp(-(x_rotated**2 + (gamma**2)*(y_rotated**2))/(2*sigma**2)))
            phase = (2*np.pi*x_rotated/lamb)+psi
            sinusoid = np.cos(phase)   # Real Gabor Filter
            kernel[i,j] = gaussian*sinusoid
    return kernel

def apply_filter(kernel,img,kernel_size):
    output_ht = (img.shape[0]-kernel_size + 0)//1 + 1
    output_wd = (img.shape[1]-kernel_size + 0)//1 + 1
    output = np.zeros((output_ht,output_wd))
    for i in range(kernel_size//2,img.shape[0]-kernel_size//2):
        for j in range(kernel_size//2,img.shape[1]-kernel_size//2):
            neighborhood = img[i - kernel_size//2 : i + kernel_size//2 + 1, j - kernel_size//2 : j + kernel_size//2 + 1]
                
            real_part = np.sum(neighborhood * np.real(kernel))
                

            output[i-kernel_size//2,j-kernel_size//2] = real_part 
    
    return output

def read_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    #print(df)
    classes_list = df['emotion']
    if csv_name == "src\\data\\fer2013.csv":
        classes_names = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
    elif csv_name == "src\\data\\ckextended.csv":
        classes_names = ['anger','disgust','fear','happiness','sadness','surprise','neutral','contempt']                
    img_arrays = df['pixels']
    # print(type(img_arrays[0]))   # ---->  returns a str
    usage = df['Usage']   # ----> Denotes whether a certain image will be used for Training or Testing

    return classes_list,classes_names,img_arrays,usage
    


def Gabor_filter_data():
    kernel_size = 11
    sigmas = [1.5]  # scales
    lambdas = [4,8]    # wavelengths
    thetas = [k * np.pi / 8 for k in range(8)] # orientations
    gamma = 0.5
    psi = 0
    # There will be 1(sigmas)*2(lambdas)*8(thetas) = 1*2*8 = 16 kernels
    no_of_filters = len(sigmas)*len(lambdas)*len(thetas)
    return kernel_size,sigmas,lambdas,thetas,gamma,psi,no_of_filters

def main():
    
    
    #print(df)
    classes_list,classes_names,img_arrays,usage = read_from_csv(csv_name=csv_name)

    imgs_list = []
    for arr in img_arrays:
        img_flattened = arr.split()    # Retrieves each pixel value as string
        img_pixels_list = np.array(img_flattened,dtype=np.uint8)    # Made an arr with those values typecasted to uint8
        final_img = img_pixels_list.reshape(48,48)  # According to dataset, each image is 48*48 hence the array is reshaped
        imgs_list.append(final_img)   # imgs_list now have proper image arrays which can be used
 

    imgs_list = np.array(imgs_list)
    print(f"Input images array's shape = {imgs_list.shape}") # 920 images of shape 48*48. So 920*48*48
    ###################################### Applying Gabor Filter ###################################### 
    kernel_size,sigmas,lambdas,thetas,gamma,psi,no_of_filters = Gabor_filter_data()
    kernels = []
    visual_kernels = np.ndarray((len(sigmas),len(lambdas),len(thetas),kernel_size,kernel_size),dtype=float)   # Will be used for visualising kernels
    for i in range(len(sigmas)):
        for j in range(len(lambdas)):
            for k in range(len(thetas)):
                kernel = filter_def(size=kernel_size,lamb=lambdas[j],gamma=gamma,sigma=sigmas[i],psi=psi,theta=thetas[k])
                # Remove the DC bias from kernel
                kernel -= kernel.mean()
                # Normalise the kernel to unit energy. 
                kernel /= (np.linalg.norm(kernel) + 1e-6)
                # Different lambdas,sigmas can give different responses in terms of energy
                # Large sigma,Large lambda filters may give large response value and so classifier may think these filters are more important
                # when in reality, they’re just numerically larger.

                # After normalization:  1) A strong response means image matches that frequency/orientation
                # Not “this filter just has higher amplitude”

                kernels.append(kernel)
                visual_kernels[i][j][k] = kernel
                
    # # Gabor Filter Bank Visualisation  (uncomment to see the kernels)
    # for i in range(len(sigmas)):
    #     for j in range(len(lambdas)):
    #         fig,ax = plt.subplots(1,len(thetas))
    #         for k in range(len(thetas)):
    #             ax[k].imshow(visual_kernels[i,j,k],cmap='gray')
    #             ax[k].set_title(f"Theta={thetas[k]:.3f}")
    #             ax[k].axis("off")
    #         plt.suptitle(f"Kernels for sigma={sigmas[i]} and lambda={lambdas[j]}")
    #         plt.tight_layout()
    #         plt.show()


    # Apply the filters
    outputs = []
    for img in imgs_list:
        for k in kernels:
            output  = apply_filter(kernel=k,img=img,kernel_size=kernel_size)
            outputs.append(output)

    outputs = np.array(outputs)  # Shape is 14720*38*38 as 920(no of ip imgs)*16(no of kernels) = 14720
    print(outputs.shape)
    outputs = outputs.reshape((imgs_list.shape[0],no_of_filters,outputs.shape[1],outputs.shape[2]))

    print(f"Output shape - {outputs.shape}")
    if csv_name == "data\\fer2013.csv":
        np.save("src\\data\\outputs_fer.npy",outputs)
    elif csv_name == "src\\data\\ckextended.csv":
        np.save("src\\data\\outputs_ck.npy",outputs)
    print("Gabor Filter Outputs Successfully Saved")

if __name__ == "__main__":
    main()