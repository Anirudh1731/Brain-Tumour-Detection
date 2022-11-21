import matplotlib.pyplot as plt
import cv2
import numpy as np



# #image reading
# image=cv2.imread('C:/Users/Anirudh/OneDrive/Desktop/Self Driving Car/lane1.png')
# # //returns an array with different pixel intensities
# # cv2_imshow(image)
# copy_image=np.copy(image)
# gray=cv2.cvtColor(copy_image,cv2.COLOR_RGB2GRAY)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)

# #gaussian blur
# filtered_image=cv2.GaussianBlur(gray,(3,3),0) #using a filter to remove the noise , here i am using a 3x3 kernal, the filtering is done by replacing values by average pixel intensities
# cv2.imshow('filtered_image',filtered_image)
# cv2.waitKey(0)


#functions
def canny(img):

  gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  # filtered_image=cv2.GaussianBlur(gray,(3,5),0)
  # filtered_image = cv2.blur(img,(5,5))
  filtered_image=cv2.medianBlur(img,7)
  # filtered_image=cv2.bilateralFilter(img,9,75,75)
  edge=cv2.Canny(filtered_image,60,150)#using canny for edge detection which searches for gradient using derivative calcultaion, if the intensity is greater than 180 it is accepted and if smaller than 60 not accepted
  return edge
def roi(img):
  h=img.shape[0]
  w=img.shape[1]
  # triangle=np.array([[0,1000],[w/2,h/2],[w,850]],np.int32)
  # pts = triangle.reshape((-1, 1, 2))
  # polygon = np.array([(0,h), (w,h), (w/2,h/2)])
  roi = np.array([(img.shape[1]//3*2, img.shape[0]//1.7), (img.shape[1]//3, img.shape[0]//1.7), (0,img.shape[0]-25), (0,img.shape[0]), (img.shape[1],img.shape[0]), (img.shape[1],img.shape[0]-25)])
  mask=np.zeros_like(img)
  cv2.fillPoly(mask, np.int32([roi]), 255)
  masked_image=cv2.bitwise_and(img,mask) #and operation to mask
  return masked_image
def display(img,lines):
  lined_image=np.zeros_like(img)
  if lines is not None:
    for line in lines:
      x1,y1,x2,y2=line.reshape(4)
      cv2.line(lined_image,(x1,y1),(x2,y2),(0,255,0),10)
  return lined_image  

cap=cv2.VideoCapture("results1.avi")
out = cv2.VideoWriter('project3.avi',0, 40,(640,342))
while(cap.isOpened()):

  _,frame=cap.read()
  canny_image=canny(frame)
  # plt.imshow(canny)
# print(canny.shape)
  # cv2.imshow('canny',canny)
  # if cv2.waitKey(1) & 0xff==ord('q'):
  #   break

# plt.show()
# plt.imshow(roi(canny))
  roi_image=roi(canny_image)
  cv2.imshow('roi',roi_image)
  if cv2.waitKey(1) & 0xff==ord('q'):
    break
# cv2_imshow(roi_image)
  # plt.imshow(roi_image)
  line=cv2.HoughLinesP(roi_image,1,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=1000) #here 1 is the rho precision and pi/180 is angle precision of the accumulator larger the value the less precise lines will be made
#here min length is the minimum length for which that line must be considered, max gap means the max gap to which the lines must be joined
# avg_line=average_intercept_and_slope(image,line)
  lined_image=display(frame,line)
# cv2_imshow(lined_image)
  req_image=cv2.addWeighted(frame,0.8,lined_image,1,1)
  cv2.imshow('video',req_image)
  # out.write(req_image)
  # # print(frame.shape[0],frame.shape[1])
  

  if cv2.waitKey(1) & 0xff==ord('q'):
    break
out.release()
cv2.destroyAllWindows()
