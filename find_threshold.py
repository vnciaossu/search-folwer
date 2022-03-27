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
		return hist


def search_Flower_Dir ():
    Name_class = []
    feat_db = []
    Path_imgDB = []
    Feat_Color = ExtractFeature_ColorDescriptor((8, 12, 3))
    f = open("Data/database/database_Flower.json", 'r')
    datajson = json.load(f)
    for i in range(len(datajson)):
        Name_class.append(datajson[i]["Class"])
        feat_db.append(datajson[i]["feat"])
        Path_imgDB.append(datajson[i]["path_img"])
    path = "Data/Testing/"
    array_threshold =[]
    for folder in os.listdir(path):
        folpath = os.path.join(path, folder)
        for fi_img in os.listdir(folpath):
            path_img = folpath +"/" + fi_img
            image = cv2.imread(path_img)
            img_resize = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
            if img_resize is None:
                return None
            feats = Feat_Color.describe(img_resize)
            feature =[]
            for feat in feats:
                if str(feat)== "0.0":
                    feature.append(0)
                else:
                    feature.append(float(feat))
            similarity_sort, indexs_sort = compare_cosine_similarity(feature, feat_db)
            array_threshold.append(similarity_sort[0])
    total_theshold=0
    for i in range(len(array_threshold)):
        total_theshold += array_threshold[i]
    median_thres = total_theshold/len(array_threshold)
    print("TB: ", median_thres)
    print ("max: ", max(array_threshold))
    print("min: ", min(array_threshold))


if __name__ == '__main__':
	search_Flower_Dir()