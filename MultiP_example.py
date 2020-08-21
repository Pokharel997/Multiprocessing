import cv2
import os
import timeit
from multiprocessing import Process

folder = "image_sample/"
folder_rotate = "Preprocessed_images/rotated_images/"
folder_croped = "Preprocessed_images/croped_images/"
folder_rotate_multip = "Preprocessed_images/rotated_multip/"
folder_croped_multip = "Preprocessed_images/cropped_multip/"


def Image_rotate_crop(filename, folderRotate, folderCrop, filename_rotate, filename_crop):
    image = cv2.imread(os.path.join(folder, filename))
    if image is not None:
        #resizing images
        h1=600
        w1=400
        dimension = (w1, h1) 
        img = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)

        #rotating images by 180 degree from center
        center = (w1 / 2, h1 / 2)
        Matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated_image = cv2.warpAffine(img, Matrix, (w1, h1))
        cv2.imwrite(os.path.join(folderRotate,filename_rotate), rotated_image)

        #cropping images
        cropped_image = img[130:400, 80:280]
        cv2.imwrite(os.path.join(folderCrop, filename_crop), cropped_image)


start_multip = timeit.default_timer()
processes = []
for filename in os.listdir(folder):
    filename_rotate_multip = filename[0:-4]+"-rotate_multip.jpg"
    filename_crop_multip = filename[0:-4]+"-crop_multip.jpg"
    process = Process(target = Image_rotate_crop, args = (filename, folder_rotate_multip, folder_croped_multip, filename_rotate_multip, filename_crop_multip))
    processes.append(process)

    process.start()
stop_multip = timeit.default_timer()

print('Time taken for multiprocess method: ', stop_multip - start_multip)

start_normal = timeit.default_timer()
for filename in os.listdir(folder):
    filename1 = filename[0:-4]+"-rotate.jpg"
    filename2 = filename[0:-4]+"-crop.jpg"
    Image_rotate_crop(filename, folder_rotate, folder_croped, filename1, filename2)
stop_normal = timeit.default_timer()

print('Time taken for normal method: ', stop_normal - start_normal)

cv2.waitKey(0)
cv2.destroyAllWindows() 

