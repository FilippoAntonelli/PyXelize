import numpy as np
import cv2
import random
import time
from face_tracking import FaceTracker
PIXELSIZE=10

def resizeImg(img,pixel_size):
    h,w,c=img.shape
    w_to_remove=w%pixel_size
    h_to_remove=h%pixel_size
    w_start=int(w_to_remove/2)
    w_end=w-w_start if (w_to_remove%2==0) else w-(w_start+1)
    h_start=int(h_to_remove/2)
    h_end= h-h_start if (h_to_remove%2==0) else h-(h_start+1)
    img=img[h_start:h_end,w_start:w_end,:]
    return img

def pixelize_colored(img,pixel_size):
    img=resizeImg(img,pixel_size)
    h,w,c=img.shape
    for wi in range (0,w,pixel_size):
        for hi in range (0,h,pixel_size):
            temp_img=img[hi:(hi+pixel_size),wi:(wi+pixel_size),:]
            b = np.mean(temp_img[:,:,0])
            g = np.mean(temp_img[:,:,1])
            r = np.mean(temp_img[:,:,2])
            temp_img[:,:,0]=b
            temp_img[:,:,1]=g
            temp_img[:,:,2]=r
    return img

def pixelize_colored_opt(img,pixel_size):
    img=resizeImg(img,pixel_size)
    h,w,c=img.shape
    small=cv2.resize(img,(int(w/pixel_size),int(h/pixel_size)),interpolation=cv2.INTER_LINEAR)
    result = cv2.resize(small,(w,h),interpolation=cv2.INTER_NEAREST)
    return result

def pixelize_grayscale(img,pixel_size):	
    img=resizeImg(img,pixel_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w=img.shape
    for wi in range (0,w,pixel_size):
        for hi in range (0,h,pixel_size):
            temp_img=img[hi:(hi+pixel_size),wi:(wi+pixel_size)]
            c = np.mean(temp_img[:,:])
            temp_img[:,:]=c
    return img 


def pixelize_dot(img,pixel_size):
    circle_img= np.ones(img.shape,dtype=np.uint8)
    img=resizeImg(img,pixel_size)
    h,w,c=img.shape
    for wi in range (0,w,pixel_size):
        for hi in range (0,h,pixel_size):
            temp_img=img[hi:(hi+pixel_size),wi:(wi+pixel_size),:]
            b = np.mean(temp_img[:,:,0])
            g = np.mean(temp_img[:,:,1])
            r = np.mean(temp_img[:,:,2])
            cv2.circle(circle_img,(wi+int(pixel_size/2),hi+int(pixel_size/2)),int(pixel_size/2),(b,g,r),-1)
    return circle_img

def pixelize_randomized(img,pixel_size):
    img=pixelize_colored(img,pixel_size)
    randomized_delta=np.random.randint(-20,20,img.shape,int)
    print(randomized_delta.shape)
    img=img+randomized_delta
    img=np.clip(img,0,255)
    return img.astype('uint8')

def pixelize_ascii(img,pixel_size):
    density='.\'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$ '
    #density=' .:-=+*#%@'
    img=resizeImg(img,pixel_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ascii_img= np.zeros(img.shape,dtype=np.uint8)
    h,w=img.shape
    for wi in range (0,w,pixel_size):
        for hi in range (0,h,pixel_size):
            temp_img=img[hi:(hi+pixel_size),wi:(wi+pixel_size)]
            c = np.mean(temp_img[:,:])
            char = density[int((c*69)/255)]
            cv2.putText(ascii_img,char,(wi,hi+pixel_size),cv2.FONT_HERSHEY_SIMPLEX,0.4,color=c,thickness=1)
    return ascii_img


def img_to_ascii(img,pixel_size):
    #density='.\'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
    density=' .:-=+*#%@'
    img=resizeImg(img,pixel_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w=img.shape
    for hi in range (0,h,pixel_size):
        for wi in range (0,w,pixel_size):
            temp_img=img[hi:(hi+pixel_size),wi:(wi+pixel_size)]
            c = np.mean(temp_img[:,:])
            char = density[int((c*9)/255)]
            #char = density[int((c*69)/255)]
            print(char,end='')
            #temp_img[:,:]=c
        print('')
    return img    
def negative(img):
    return 255-img
def blur(img,radius):
    img=resizeImg(img,radius)
    h,w,c=img.shape
    for hi in range(0,h,radius):
        for wi in range (0,w,radius):
            temp_img=img[hi:(hi+radius),wi:(wi+radius)]
            print (temp_img.shape)
            print(img.shape)
            img[hi,wi,0]=np.mean(temp_img[:,:,0])
            img[hi,wi,1]=np.mean(temp_img[:,:,1])
            img[hi,wi,2]=np.mean(temp_img[:,:,2])
    return img


faceTracker=FaceTracker()
vid = cv2.VideoCapture(0)
while(True):
    one_loop_time=time.time()
    ret, frame = vid.read()
    # Display the resulting frame
    #frame=pixelize_ascii(negative(frame),PIXELSIZE)
    #frame=pixelize_colored_opt(frame,PIXELSIZE)
    #frame=blur(frame,PIXELSIZE)
    #frame=negative(frame)
    faces = faceTracker.trackFaces(frame,drawBoxes=False)
    for (x, y, w, h) in faces:
        np.random.shuffle(frame[y:y+h,x:x+w,:])
    one_loop_time=time.time()-one_loop_time
    fps=str(round(1/one_loop_time,2))
    cv2.putText(frame,fps,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0,0,255),thickness=2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()


#img =cv2.imread('homer.jpg')
#img=img_to_ascii(img,PIXELSIZE)
#cv2.imwrite('giovanni_muciaccia_cropped.jpg',img)
