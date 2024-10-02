import io
import os
import sys
import cv2
import csv
import math
import time
import numpy as np

import requests
from PIL import Image
from io import BytesIO
from httplib2 import Http
from threading import Thread

class Map(object):
    def __init__(self, coors, zoom, size=(512, 512), load_size=(1024, 1024), zoommin=11, zoommax=18, ifdl=False, fps=30):
        sys.path.append('.')
        self.size = size
        self.zoom = zoom
        self.coors = coors
        self.lsize = load_size

        self.fps = fps
        self.nmark = 0
        self.load = False
        self.flagd = ifdl
        self.errload = 0
        self.tbfload = 0.5
        self.zoommin = zoommin
        self.zoommax = zoommax
        self.dlat = self.dlon = 0
        self.dlatn = self.dlonn = 0

        self.MAP = {"map": np.zeros((load_size[0], load_size[1], 3), dtype=np.int8), "img": np.zeros((load_size[0], load_size[1], 3), dtype=np.int8),
                    "start": [0, 0], "end": [1, 1], "zoom": 0, "marks": {}, "nmarks": {}}
        # self.update(coors, zoom)

        pad = np.zeros((256, 256, 3))
        cv2.putText(pad, "NO DATA", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imwrite("./tilefile/warning.jpg", pad)
        self.pad = np.concatenate([np.concatenate([pad for i in range(0, size[0]//256)], axis=1) for i in range(0, size[0]//256)], axis=0)

        Thread(target=self.THREAD_APPROCHEDIS, daemon=True).start()
        Thread(target=self.THREAD_CHECKAVAILIBLE, daemon=True).start()

    @staticmethod
    def latlon2tile(lat_deg, lon_deg, zoom):
        n = math.pow(2, zoom)
        tilex = n * ((lon_deg + 180.0) / 360.0)
        tiley = n * (1 - (math.log(math.tan((lat_deg * math.pi) / 180.0) + 1.0 / math.cos((lat_deg * math.pi) / 180.0)) / math.pi)) / 2
        return tilex, tiley

    @staticmethod
    def tile2latlon(tilex, tiley, zoom):
        n = math.pow(2, zoom)
        lon_deg = tilex / n * 360.0 - 180.0
        lat_deg = math.atan(math.sinh(math.pi * (1 - 2 * tiley / n))) * 180.0 / math.pi
        return lat_deg, lon_deg

    @staticmethod
    def pixel2lonlat(pixel, zoom):
        n = pixel / 256 * math.pow(2, zoom)
        return 85.0511 * 2 / n, 180 * 2 / n

    def update(self, coors, zoom):
        lat, lon = coors
        self.load = False
        tx, ty = self.latlon2tile(lat, lon, zoom)
        itx, ity = math.floor(tx), math.floor(ty)  # DO NOT USE INT

        ilat, ilon = self.tile2latlon(itx, ity, zoom)
        self.dlatn, self.dlonn = (lat - ilat) / 2, (lon - ilon) / 2

        iimage = []
        for i in range(itx - math.ceil(self.lsize[0] / 2 / 256) + math.ceil(abs(tx - itx)), itx + math.ceil(self.lsize[0] / 2 / 256) + math.ceil(abs(tx - itx))):
            jimage = []
            for j in range(ity - math.ceil(self.lsize[1] / 2 / 256) + math.ceil(abs(ty - ity)), ity + math.ceil(self.lsize[1] / 2 / 256) + math.ceil(abs(ty - ity))):
                path = './TileQ/%s/%s_%s.jpg' % (zoom, i, j)
                if os.path.isfile(path): jimage.append(cv2.imread(path))
                else:
                    if self.flagd: jimage.append(self.download(zoom, i, j))
                    else: jimage.append(cv2.imread("./TileQ/warning.jpg"))
            iimage.append(np.concatenate([img for img in jimage], axis=0))

        img = np.concatenate([img for img in iimage], axis=1)
        slat, slon = self.tile2latlon(itx - math.ceil(self.lsize[0] / 2 / 256) + math.ceil(abs(tx - itx)), ity - math.ceil(self.lsize[1] / 2 / 256) + math.ceil(abs(ty - ity)), zoom)
        elat, elon = self.tile2latlon(itx + math.ceil(self.lsize[0] / 2 / 256) + math.ceil(abs(tx - itx)), ity + math.ceil(self.lsize[1] / 2 / 256) + math.ceil(abs(ty - ity)), zoom)

        self.MAP["img"] = img
        self.MAP["map"] = img
        self.MAP["start"] = [slat, slon]
        self.MAP["end"] = [elat, elon]
        self.MAP["zoom"] = zoom

    def center(self, coors):
        # if self.zoom != self.MAP["zoom"]: self.update(coors, self.zoom)
        self.coors = coors
        h, w, ch = self.MAP["img"].shape

        tx, ty = self.latlon2pixel((coors[0] - self.dlat), (coors[1] - self.dlon), w, h)
        sx, sy = self.latlon2pixel(self.MAP["start"][0] - self.dlat, self.MAP["start"][1] - self.dlon, w, h)
        ex, ey = self.latlon2pixel(self.MAP["end"][0] - self.dlat, self.MAP["end"][1] - self.dlon, w, h)

        cx, cy = int(w * (tx - sx) / (ex - sx)), int(h * (ty - sy) / (ey - sy))
        sx, sy, ex, ey = cx-self.size[0]//2, cy-self.size[1]//2, cx+self.size[0]//2, cy+self.size[1]//2
        dsx, dsy, dex, dey = 0, 0, 0, 0
        # print(sx, sy, ex, ey, "START")
        if sx < 0:
            dsx = sx - 0  # - (sx - 0) // self.size[0] * self.size[0]
            sx = 0
        elif sx > w:
            dsx = w - sx  # - (w - sx) // self.size[0] * self.size[0]
            sx = w
        if sy < 0:
            dsy = sy - 0  # - (sy - 0) // self.size[1] * self.size[1]
            sy = 0
        elif sy > h:
            dsy = h - sy  # - (h - sy) // self.size[1] * self.size[1]
            sy = h
        if ex < 0:
            dex = ex - 0  # - (ex - 0) // self.size[0] * self.size[0]
            ex = 0
        elif ex > w:
            dex = w - ex  # - (w - ex) // self.size[0] * self.size[0]
            ex = w
        if ey < 0:
            dey = ey - 0  # - (ey - 0) // self.size[1] * self.size[1]
            ey = 0
        elif ey > h:
            dey = h - ey  # - (h - ey) // self.size[1] * self.size[1]
            ey = h
        if abs(dsx) + abs(dex) > self.size[0]:
            dsx = self.size[0]
            dex = 0
        if abs(dsy) + abs(dey) > self.size[1]:
            dsy = self.size[1]
            dey = 0

        if sx != 0 or ex != 0 or sy != 0 or ey != 0: self.errload += 1
        else: self.errload = 0

        '''
                img = self.MAP["map"][cy-self.size[1]//2:cy+self.size[1]//2, cx-self.size[0]//2:cx+self.size[0]//2]
                h, w, ch = img.shape
                print(h, w, ch)
                if (w, h) != self.size:
                    rmmark = self.MAP["data"].copy()
                    self.update(coors, self.zoom)
                    h, w, ch = self.MAP["img"].shape
                    tx, ty = self.latlon2tile(coors[0] - self.dlat, coors[1] - self.dlon, self.MAP["zoom"])
                    sx, sy = self.latlon2tile(self.MAP["start"][0], self.MAP["start"][1], self.MAP["zoom"])
                    ex, ey = self.latlon2tile(self.MAP["end"][0], self.MAP["end"][1], self.MAP["zoom"])
                    cx, cy = int(w * (tx - sx) / (ex - sx)), int(h * (ty - sy) / (ey - sy))
                    self.clearmark()
                    for _, mark in rmmark.items(): self.addmark(mark["coors"], mark["radius"], mark["color"], mark["data"])
                    img = self.MAP["map"][cy - self.size[1] // 2:cy + self.size[1] // 2, cx - self.size[0] // 2:cx + self.size[0] // 2]
                    # print(tx, ty, sx, sy, ex, ey)
        '''

        # print(sx, sy, ex, ey, "END")
        # print(dsx, dex, dsy, dey, "PAD")
        img = self.MAP["map"][sy:ey, sx:ex]
        img = cv2.copyMakeBorder(img, abs(dsy), abs(dey), abs(dsx), abs(dex), cv2.BORDER_CONSTANT, value=(0, 0, 0))

        if self.load:
            mask = np.resize(img, (self.size[0] // 256, self.size[1] // 256, 3))
            mask = np.resize(-np.where(np.where(mask < .1, -1, mask) > .1, 0, np.where(mask < .1, -1, mask)), (self.size[0], self.size[1], 3))
            mask = -np.where(np.where(img < .1, -1, img) > .1, 0, np.where(img < .1, -1, img)) * mask * self.pad
            return np.add(img.astype(np.uint8), mask.astype(np.uint8))
        else:
            return img

    def checkavailible(self, coors):
        if self.zoom != self.MAP["zoom"]: self.update(coors, self.zoom)
        h, w, ch = self.MAP["img"].shape

        tx, ty = self.latlon2pixel((coors[0] - self.dlat), (coors[1] - self.dlon), w, h)
        sx, sy = self.latlon2pixel(self.MAP["start"][0] - self.dlat, self.MAP["start"][1] - self.dlon, w, h)
        ex, ey = self.latlon2pixel(self.MAP["end"][0] - self.dlat, self.MAP["end"][1] - self.dlon, w, h)

        cx, cy = int(w * (tx - sx) / (ex - sx)), int(h * (ty - sy) / (ey - sy))
        h, w, ch = self.MAP["map"][cy - self.size[1]:cy + self.size[1], cx - self.size[0]:cx + self.size[0]].shape
        if (w, h) != (self.size[0] * 2, self.size[1] * 2):
            # rmmark = self.MAP["data"].copy()
            self.update(coors, self.zoom)
            # self.clearmark()
            # for _, mark in rmmark.items():
            #     self.addmark(mark["coors"], mark["radius"], mark["color"], mark["data"])
            # self.plot()

    def addmark(self, coors, radius=3, color=(0, 0, 0), data="", name=""):
        h, w, ch = self.MAP["img"].shape

        tx, ty = self.latlon2pixel((coors[0] - self.dlat), (coors[1] - self.dlon), w, h)
        sx, sy = self.latlon2pixel(self.MAP["start"][0] - self.dlat, self.MAP["start"][1] - self.dlon, w, h)
        ex, ey = self.latlon2pixel(self.MAP["end"][0] - self.dlat, self.MAP["end"][1] - self.dlon, w, h)

        cx, cy = int(w * (tx - sx) / (ex - sx)), int(h * (ty - sy) / (ey - sy))
        if len(name) > 0:
            self.MAP["nmarks"][str(name)] = {"coors": coors, "x": cx, "y": cy, "radius": radius, "color": color, "data": data}
        else:
            self.MAP["marks"]["point%x"%int(self.nmark)] = {"coors": coors, "x": cx, "y": cy, "radius": radius, "color": color, "data": data}
            self.nmark += 1

    def download(self, zoom, tilex, tiley):
        # url = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/%s/%s/%s.png" % (zoom, tiley, tilex)
        url = "https://mt1.google.com/vt/lyrs=y&x=%s&y=%s&z=%s" % (tilex, tiley, zoom)

        print("DOWNLOADING: " + url)
        try:
            self.load = True
            req = Http()
            datafile = BytesIO(req.request(url)[1])
            datafile.seek(0)

            img = cv2.cvtColor(np.array(Image.open(datafile)), cv2.COLOR_RGB2BGR)
            cv2.imwrite('./TileQ/%s/%s_%s.jpg' % (zoom, tilex, tiley), img)
            req.close()

            return img
        except Exception:
            print("DOWNLOAD FAIL")
            return cv2.imread("./TileQ/warning.jpg")

    @staticmethod
    def latlon2pixel(lat_deg, lon_deg, w, h):
        x = (lon_deg + 180) * (h / 360)
        latRad = lat_deg * math.pi / 180
        mercN = math.log(math.tan((math.pi / 4) + (latRad / 2)))
        y = (h / 2) - (w * mercN / (2 * math.pi))
        return x, y

    def zoomin(self):
        if self.zoom <= self.zoommax: self.zoom += 1

    def zoomout(self):
        if self.zoom >= self.zoommin: self.zoom -= 1

    def clearmark(self):
        self.nmark = 0
        self.MAP["marks"] = {}
        self.MAP["map"] = self.MAP["img"].copy()

    def plot(self):
        for _, mark in self.MAP["marks"].copy().items():
            cv2.circle(self.MAP["map"], (mark["x"], mark["y"]), mark["radius"], mark["color"], 3, cv2.LINE_AA)
            if len(mark["data"]) > 0:
                cv2.putText(self.MAP["map"], str(mark["data"]), (mark["x"]+mark["radius"]+5, mark["y"]), cv2.FONT_HERSHEY_SIMPLEX, mark["radius"]/10, mark["color"], 1, cv2.LINE_AA)

        for _, mark in self.MAP["nmarks"].copy().items():
            cv2.circle(self.MAP["map"], (mark["x"], mark["y"]), mark["radius"], mark["color"], 3, cv2.LINE_AA)
            if len(mark["data"]) > 0:
                cv2.putText(self.MAP["map"], str(mark["data"]), (mark["x"]+mark["radius"]+5, mark["y"]), cv2.FONT_HERSHEY_SIMPLEX, mark["radius"]/10, mark["color"], 1, cv2.LINE_AA)

    def replot(self):
        rmmarks = self.MAP["marks"].copy()
        rmnmarks = self.MAP["nmarks"].copy()
        self.clearmark()
        for _, mark in rmmarks.items():
            self.addmark(mark["coors"], mark["radius"], mark["color"], mark["data"])
        for _, mark in rmnmarks.items():
            self.addmark(mark["coors"], mark["radius"], mark["color"], _)
        self.plot()

    def approchedis(self):
        if self.dlat == 0:
            self.dlat = self.dlatn
        if self.dlon == 0:
            self.dlon = self.dlonn

        if self.dlat != self.dlatn:
            if self.dlat < self.dlatn:
                self.dlat += (self.dlatn - self.dlat) / self.fps
            else:
                self.dlat -= (self.dlat - self.dlatn) / self.fps

        if self.dlon != self.dlonn:
            if self.dlon < self.dlonn:
                self.dlon += (self.dlonn - self.dlon) / self.fps
            else:
                self.dlon -= (self.dlon - self.dlonn) / self.fps
    def THREAD_CHECKAVAILIBLE(self):
        while True:
            self.checkavailible(self.coors)
            if self.errload <= self.fps: self.tbfload += 0.1
            if self.errload >= self.fps and self.tbfload >= 0.1:
                self.tbfload -= 0.1
                self.errload = 0
            time.sleep(self.tbfload)

    def THREAD_APPROCHEDIS(self):
        while True:
            self.approchedis()
            time.sleep(0.1)


if __name__ == '__main__':
    lat, lon = (LON, LAT)
    main = Map((lat, lon), 17, (512, 512), (2048, 2048), ifdl=False)   # (COORDINATES, ZOOM-START, SHOW_MAP_SIZE, LOAD_MAP_SIZE, DO_DOWNLOAD?)

    # EXAMPLES OF ADDING A MARK:
    # main.addmark((LON, LAT), data="CAR", name="CAR1")
    # main.addmark((LON, LAT), data="CAR", name="CAR2")
    # main.addmark((LON, LAT), data="CAR", name="CAR3")

    while True:
        with open("./test.csv") as file:  # USE CUSTOM GPS DATA
            coors = csv.reader(file, delimiter="\n")
            for coor in coors:
                lat, lon = float(coor[0].split(",")[0]), float(coor[0].split(",")[1])
                main.clearmark()
                main.replot()
                main.addmark((lat, lon), data="CURRENT LOCATION")
                main.plot()
                # img = main.center((lat+np.random.uniform(low=-0.00001, high=0.00001), lon+np.random.uniform(low=-0.00001, high=0.00001)))
                
                img = main.center((lat, lon))
                cv2.imshow(" ", img)
                cv2.waitKey(10)
                
            main.zoomout()
            # main.zoomin()
    # _ = map.update()
    # cv2.imshow("", _)
