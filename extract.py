from collections import Counter
from typing import List
from PIL import Image
import PIL
import math
from sklearn.cluster import DBSCAN, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

class ColorExtractor(object):

    def __init__(self, COLOR_LIMIT=10, DIST_THRESHOLD=50, SIZE_LIMIT = 500):
        self.COLOR_LIMIT = COLOR_LIMIT # Max number of colors to display
        self.DIST_THRESHOLD = DIST_THRESHOLD
        self.SIZE_LIMIT = SIZE_LIMIT

    def get_pixel_freq(self, image: Image) -> Counter:
        return Counter(image.getdata())

        
    def get_pixel_list(self, image) -> list:
        pixel_list = []

        row, col = image.size
        for x in range(row):
            for y in range(col):

                r, g, b = image.getpixel((x,y))

                pixel_list.append((r,g,b))

        return pixel_list
 
    def get_dominant_colors_clustering(self, img, model='kmeans'):

        img = self.preprocess_image(img)
        
        color_list = img.getdata()

        X = np.array(color_list)

        # define the model
        if model == "DBSCAN":
            model = DBSCAN(eps=0.30, min_samples=9)
            print("fitting...")
            model.fit_predict(X)

        elif model == "BIRCH":
            model = Birch(threshold=0.01, n_clusters=2)
            print("fitting...")
            model.fit(X)
            print("predicting...")
            yhat = model.predict(X)

        elif model == "gaussian":
            model = GaussianMixture(n_components=5)
            print("fitting...")
            model.fit(X)
            print("predicting...")
            yhat = model.predict(X)

        elif model == "kmeans":
            model = MiniBatchKMeans(n_clusters=5)
            print("fitting...")
            model.fit(X)
            print("predicting...")
            yhat = model.predict(X)

        clusters = np.unique(yhat)

        dominant_colors = []

        for cluster in clusters:

            row_ix = np.where(yhat == cluster)

            # print(cluster)

            r, g, b = round(np.average(X[row_ix, 0])), round(np.average(X[row_ix, 1])), round(np.average(X[row_ix, 2]))

            dominant_colors.append((r,g,b))
            # print("(",r,g,b,")")
            # print(self.rgb_to_hex((r,g,b)))

        return dominant_colors

    def get_dominant_colors(self, color_dict: Counter):

        for c1 in list(color_dict.keys()):

            for c2 in list(color_dict.keys()):

                dist = math.dist(c1, c2)

                if dist < self.DIST_THRESHOLD and dist != 0:

                    weighted_color = self.get_weighted_average(c1, color_dict[c1], c2, color_dict[c2])

                    color_dict[weighted_color] += color_dict[c1] + color_dict[c2]
                    
                    del color_dict[c1]

                    del color_dict[c2]

                    break
        
        return color_dict

    def get_dominant_colors_hist(self, img: Image):

        hist = img.histogram()

        r, g, b = hist[0:256], hist[256: 256*2], hist[256*2: 256*3]

        print(len(r), len(g), len(b))

        color_dict = self.get_pixel_freq(img)
        
    def get_color_pallete(self, color_list):
        pass

    def get_color_pallete_image(self, colors: list):

        new_img = Image.new(mode='RGB', size=(100*len(colors), 200), color='white') 

        for idx, color in enumerate(colors):
            
            new_img.paste(color, (idx*100, 100, idx*100+100, 200))


        new_img.show()

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb 

    def hex_to_rgb(self, hex):
        hex = hex[1:]
        return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

    def get_weighted_average(self, color1, freq1, color2, freq2):

        total_freq = freq1+freq2

        color1 = list(color1)
        color2 = list(color2)
        weighted_color = []

        for c1, c2 in zip(color1, color2):

            c1 *= (freq1/total_freq)

            c2 *= (freq2/total_freq)

            weighted_color.append(round(c1+c2))

        return tuple(weighted_color)

    def preprocess_image(self, image):

        max_side = max(image.width, image.height)

        if max_side > self.SIZE_LIMIT:
            scale_factor = max_side / self.SIZE_LIMIT
            new_image = image.transform((int(image.width / scale_factor), int(image.height // scale_factor)), Image.EXTENT, data =[0, 0,  image.width , image.height])
        
        new_image.show()
        return new_image

    # row, col = i.size

    # for x in range(row):
    #     for y in range(col):

    #         r, g, b = i.getpixel((x,y))

    #         r *= random.random()
    #         g *= random.random() 
    #         b *= random.random()

    #         i.putpixel((x,y), (int(r), int(g), int(b)))


if __name__ == "__main__":

    img = Image.open(input("Enter path/name for image: "))

    print(img.size)
    
    ce = ColorExtractor()
    # ce.preprocess_image(img).show()
    # p_image = ce.preprocess_image(img)
    
    start = time.time()
    dom_colors = ce.get_dominant_colors_clustering(img, model='kmeans')
    print("time elapsed: ", time.time()-start)

    ce.get_color_pallete_image(dom_colors)

    img.show()
