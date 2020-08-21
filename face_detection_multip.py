import cv2
import os
from timeit import default_timer as timer
from multiprocessing import Process, cpu_count
from mtcnn.mtcnn import MTCNN

#Folder name initialization
folder = "image_sample/"
folder_rotate = "Preprocessed_images/rotated_images/"
folder_croped = "Preprocessed_images/croped_images/"
folder_rotate_multip = "Preprocessed_images/rotated_multip/"
folder_croped_multip = "Preprocessed_images/cropped_multip/"
folder_faceimg = "Preprocessed_images/face_image/"
folder_faceimg_multip = "Preprocessed_images/face_image_multip/"

#Face detection and face image crop function
def Image_face_crop(filename, folder_name, filename_faceimg):

    image = cv2.imread(os.path.join(folder, filename))
    process_id = os.getpid()
    print(f"Process Id for {filename} is {process_id}")
    detector = MTCNN()

    # This internal method detects faces in the image
    result = detector.detect_faces(image)
    if result != []:
        # Result is an array with all the bounding boxes detected.
        bounding_box = result[0]['box']
        # keypoints = result[0]['keypoints']

        cv2.rectangle(image,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                    (0,155,255),
                    2)
        cropped_image = image[bounding_box[0]+bounding_box[2]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[1]]
        cv2.imwrite(os.path.join(folder_name,filename_faceimg), cropped_image)
        print(result)

def main():
    # start_multip = timer()
    print(f'starting computations on {cpu_count()} cores')
    processes = []
    files = os.listdir(folder)
    for filename in files:
        filename_facecrop_multip = filename[0:-4]+"-facecrop_multip.jpg"
        process = Process(target = Image_face_crop, args = (filename, folder_faceimg_multip, filename_facecrop_multip))
        processes.append(process)
        process.start()

cv2.waitKey(0)
cv2.destroyAllWindows() 

if __name__ == '__main__':

    main()


