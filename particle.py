import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
def distance(x,y,start_blue_x,start_blue_y):
	return math.sqrt( math.pow((start_blue_x-x),2) + math.pow((start_blue_y-y),2) )

def calculate_blue_mean(distance_blue):
	mean = 658.6*distance_blue + 234
	return mean


#mpl.rcParams['legend.fontsize'] = 10   
#fig = plt.figure()						
#ax = fig.gca(projection='3d')			

partical_x = [random.uniform(0,1000) for x in range(1000)]
partical_y = [random.uniform(0,500) for i in range(1000)]
colors = np.random.rand(1000)


plt.ion()
#plt.figure()

len_par = len(partical_x)
start_green_x = 0
start_green_y = 81.5
start_blue_x = 0
start_blue_y = 0
#position_green_x = []
#position_green_y = []
#pop_green = []
#position_blue_x = []
#position_blue_y = []
#pop_blue = []
#pop_blue.append(float( np.exp(-1*( ((math.pow( (x[i]-mean_x),2 ) )/2) + ((math.pow( (y[j]-mean_y),2 ))/2) ) ) ) ) 
area_blue_count = []
area_green_count = []

variance = 1000
sd = math.sqrt(variance)
H_B_MIN = 109
S_B_MIN = 100
V_B_MIN = 40
H_B_MAX = 179
S_B_MAX = 255
V_B_MAX = 255

#H_B_MIN = 39
#S_B_MIN = 3
#V_B_MIN = 224
#H_B_MAX = 114
#S_B_MAX = 131
#V_B_MAX = 255

H_G_MIN = 20
S_G_MIN = 30
V_G_MIN = 0
H_G_MAX = 48
S_G_MAX = 255
V_G_MAX = 255
#partical = np.zeros((500,1000),dtype=np.int)
partical_blue_count = np.zeros((1000,),dtype=np.int)
partical_green_count = np.zeros((1000,),dtype=np.int)

blue_min = np.array([H_B_MIN,S_B_MIN,V_B_MIN],np.uint8)
blue_max = np.array([H_B_MAX,S_B_MAX,V_B_MAX],np.uint8)
green_min = np.array([H_G_MIN,S_G_MIN,V_G_MIN],np.uint8)
green_max = np.array([H_G_MAX,S_G_MAX,V_G_MAX],np.uint8)

camera_focus_length = 1.4168809800297879e+03

object_blue_length = 210 #210 mm
object_green_length = 210 #210mm
#real_object_distance = 1.4
cap = cv2.VideoCapture(0)
cap.set(3,800)
cap.set(4,450)
cap.set(15,-3)
inrange_blue_frame = 0
blur = 0
hsv = 0

#for i in range(len(partical_x)):
#	cv2.circle(partical,(partical_x[i],partical_y[i]),5,255,thickness=-1)
while(True):
	ret,frame = cap.read()
	blur = cv2.GaussianBlur(frame,(5,5),1)
	blur = cv2.GaussianBlur(blur,(3,3),0)
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	inrange_frame = cv2.inRange(hsv,blue_min,blue_max)
	inrange_frameG = cv2.inRange(hsv,green_min,green_max)

	plt.scatter(partical_x, partical_y, s=2, c=colors, alpha=0.5)

	k = cv2.waitKey(10)
	if k == 27 :
		break
	elif k == 99 :
		#print(image_blue_length,object_blue_distance)#,"<<<>>>",image_green_length,object_green_distance)
		

		inrange_blue_frame = cv2.inRange(hsv,blue_min,blue_max)
		inrange_green_frame = cv2.inRange(hsv,green_min,green_max)
		
		r,c = inrange_blue_frame.shape
		area_blue_count = 0
		area_green_count = 0
		
		partical_blue_count = np.zeros((500,),dtype=np.int)
		partical_green_count = np.zeros((500,),dtype=np.int)
		for i in range(r):
			for j in range(c):
				if inrange_blue_frame.item(i,j) > 0:
					area_blue_count+=1
				if inrange_green_frame.item(i,j) > 0:
					area_green_count+=1
		image_blue_length = np.sqrt(area_blue_count)
		image_green_length = np.sqrt(area_green_count)
		object_blue_distance = (((object_blue_length/image_blue_length)*camera_focus_length)/2)/10
		object_green_distance = (((object_green_length/image_green_length)*camera_focus_length)/2)/10
		
		#mean_blue = calculate_blue_mean(object_blue_distance)/10
		#mean_green = calculate_green_mean(object_green_distance)/10
		select_pop = []
		total_pop = 0
		tmp_x = []
		tmp_y = []
		#pop_blue = []
		#pop_green = []
		for i in range(len_par):
			dis = distance(partical_x[i],partical_y[i],start_blue_x,start_blue_y)
			dis_g = distance(partical_x[i],partical_y[i],start_green_x,start_green_y)
			#test1 = 1/(sd*math.sqrt(2*math.pi))
			#test2 = np.exp((-1)*( math.pow((dis-object_blue_distance),2)/(2*math.pow(sd,2)))) 
			
			#print(test1,test2)
			#print(dis,object_blue_distance)
			#pop.append(1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis-object_blue_distance),2)/(2*math.pow(sd,2)) ) ))
			#pop_green.append(1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis_g-object_green_distance),2)/(2*math.pow(sd,2)) ) ))
			#a = (1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis-object_blue_distance),2)/(2*math.pow(sd,2)) ) ))
			#b = (1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis_g-object_green_distance),2)/(2*math.pow(sd,2)) ) ))
			total_pop+=((1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis-object_blue_distance),2)/(2*math.pow(sd,2)) ) ))*(1*(1/(sd*math.sqrt(2*math.pi)))*np.exp((-1)*( math.pow((dis_g-object_green_distance),2)/(2*math.pow(sd,2)) ) )))

			select_pop.append(total_pop)
		#print("pop_blue",pop_blue[0],"total_pop",total_pop)
		for i in range(len_par):
			if i < 0.95*len_par :	
				select = random.uniform(0,total_pop)
				for j in range(len(select_pop)):
					if select_pop[j] >= select :
						tmp_x.append(partical_x[j])
						tmp_y.append(partical_y[j])
						partical_blue_count += 1
						break
			else:
				tmp_x.append(random.uniform(0,1000))
				tmp_y.append(random.uniform(0,500))
		partical_x = tmp_x
		partical_y = tmp_y

		print(image_blue_length,object_blue_distance)
		plt.scatter(partical_x, partical_y, s=2, c=colors, alpha=0.5)
	elif k == ord('s'):
		plt.plot(partical_blue_count)

	plt.draw()
	plt.pause(0.001)
	plt.clf()
	cv2.imshow('inRange',inrange_frame)
	cv2.imshow('inRange_GREEN',inrange_frameG)
	cv2.imshow('frame',frame)

	
#ax.legend()
cap.release()
cv2.destroyAllWindows()