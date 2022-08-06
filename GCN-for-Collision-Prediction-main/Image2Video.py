import os 
import cv2


path = "Plots"

images =  os.listdir(path)


number = []
img_array = []

for i in images:

	num = i.split(".")
	number.append(int(num[0] ))


ImageOrder = sorted(number)

cnt = 0

for img_num in range ( 0 , len(ImageOrder) , 8):

	img = ImageOrder[img_num]

	filename = path +"/" +  str(img) + ".png"
	# print(filename)
	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width,height)
	img_array.append(img)
	print( "img " +  str(cnt)  + " complete")
	cnt +=1 


out = cv2.VideoWriter('project_prodict5.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()