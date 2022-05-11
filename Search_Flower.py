import os
import json
import math
import cv2
import numpy as np
from scipy.spatial.distance import cosine

def compare_cosine_similarity ( feat_img, feat_db):
	temps_similarity = []
	for i in range(len(feat_db)):
		temp_similarity = cosine(feat_img, feat_db[i])
		temps_similarity.append(temp_similarity)

	similarity_sort = np.sort(temps_similarity) 
	indexs_sort = np.argsort(temps_similarity)
	return similarity_sort, indexs_sort

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
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])

		cv2.normalize(hist,hist)

		hist = hist.flatten()
		print("hist flatten", hist)
		return hist

def search_Flower():
	Name_class =[]
	feat_db =[]
	Path_imgDB= []
	f = open("Train/database_Flower.json", 'r')
	datajson = json.load(f)
	for i in range (len (datajson)):
		Name_class.append(datajson[i]["Class"])
		feat_db.append(datajson[i]["feat"])
		Path_imgDB.append(datajson[i]["path_img"])

	Feat_Color = ExtractFeature_ColorDescriptor((8, 12, 3))

	image = cv2.imread("./279581152_961779464419830_7933107883665669983_n.jpg")

	img_resize = cv2.resize(image, (600,600))
	if img_resize is None:
		return None
	feats = Feat_Color.describe(img_resize)
	feature =[]
	for feat in feats:
		feature.append(float(feat))
	similarity_sort, indexs_sort = compare_cosine_similarity(feature, feat_db)
	Name_class[indexs_sort[0]], Path_imgDB[indexs_sort[0]]
	image_query = cv2.imread(Path_imgDB[indexs_sort[0]])
	image_query = cv2.resize(image_query, (600,600), interpolation = cv2.INTER_AREA)
	image_show = cv2.hconcat([img_resize, image_query])
	if similarity_sort[0] >0.65:
		text = "No database -" + " Similarity: " + str(round(similarity_sort[0], 2)) + "-" + Name_class[indexs_sort[0]]
	else:
		text = "Label: " + Name_class[indexs_sort[0]] + " Similarity: " + str(similarity_sort[0])
	print("similarity_sort",similarity_sort)
	cv2.putText(img_resize,text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	cv2.imshow("Image Search", img_resize)
	cv2.waitKey(0)

if __name__ == '__main__':
	search_Flower()