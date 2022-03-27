import os
import json
import math
import cv2
import numpy as np

def resize_database_Flower():
	path_Input ="Train/"
	path_Dirsave="Train/"
	for folder in os.listdir(path_Input):
		folpath = os.path.join(path_Input, folder)
		path_save = path_Dirsave + folder + "/"
		if not os.path.exists(path_save):
			os.mkdir(path_save)
		for fi_img in os.listdir(folpath):
			path_img= os.path.join(folpath, fi_img)
			imgFlower_database = cv2.imread(path_img, cv2.IMREAD_COLOR)
			img_resize = cv2.resize(imgFlower_database, (600,600), interpolation = cv2.INTER_AREA)
			path_img_save = path_save +fi_img
			cv2.imwrite(path_img_save, img_resize)

if __name__ == '__main__':
	resize_database_Flower()