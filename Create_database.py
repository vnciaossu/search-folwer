import os
import json
import math
import cv2
import numpy as np

class ExtractFeature_ColorDescriptor:
   def __init__(self, bins):
       self.bins = bins

   def describe(self, image):
       image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
       features = []
       (h, w) = image.shape[:2]
       (cX, cY) = (int(w * 0.5), int(h * 0.5))
       segments = [(0, cX, 0, cY), (cX, w, 0, cY),(cX, w, cY, h), (0, cX, cY, h)]

       (axesX, axesY) = (int((w * 0.75) / 2), int((h * 0.75) / 2))
       ellipMask = np.zeros(image.shape[:2], dtype="uint8")
       cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

       for (startX, endX, startY, endY) in segments:
           cornerMask = np.zeros(image.shape[:2], dtype="uint8")
           cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
           cornerMask = cv2.subtract(cornerMask, ellipMask)

           hist = self.histogram(image, cornerMask)
           features.extend(hist)

       hist = self.histogram(image, ellipMask)
       features.extend(hist)
       return features

   def histogram(self, image, mask):
       hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                           [0, 180, 0, 256, 0, 256])

       cv2.normalize(hist,hist)
       hist = hist.flatten()

       return hist


def create_database_Flower():
	Feat_Color = ExtractFeature_ColorDescriptor((8, 12, 3))

	Data_json =[]
	path ="Train/"
	for folder in os.listdir(path):
		folpath = os.path.join(path, folder)
		for fi_img in os.listdir(folpath):
			path_img= os.path.join(folpath, fi_img)
			imgFlower_database = cv2.imread(path_img, cv2.IMREAD_COLOR)
			img_resize = cv2.resize(imgFlower_database, (600,600), interpolation = cv2.INTER_AREA)
			if img_resize is None:
				continue
			feats = Feat_Color.describe(img_resize)
			feature =[]
			for feat in feats:
				feature.append(float(feat))
			data ={"feat": feature, "Class": folder, "path_img": path_img}
			Data_json.append(data)

	with open("Train/database_Flower.json", "w") as outfile: 
		data = json.dumps(Data_json)
		outfile.write(data)

if __name__ == '__main__':
	create_database_Flower() 