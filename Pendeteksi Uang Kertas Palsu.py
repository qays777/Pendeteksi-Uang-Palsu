import os
import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.actionLoad_Uang.triggered.connect(self.fungsi)
        self.pushButton.clicked.connect(self.grayscale)
        self.pushButton_4.clicked.connect(self.brightness)
        self.pushButton_3.clicked.connect(self.ssim50)
        self.pushButton_2.clicked.connect(self.ssim)
        self.action2_Ribu_Rupiah.triggered.connect(self.ssim2)
        self.action5_Ribu_Rupiah.triggered.connect(self.ssim5)
        self.action10_Ribu_Rupiah.triggered.connect(self.ssim10)
        self.action20_Ribu_Rupiah.triggered.connect(self.ssim20)
        self.action50_Ribu_Rupiah.triggered.connect(self.ssim50)
        self.action100_Ribu_Rupiah.triggered.connect(self.ssim)

    def loadImage(self, flname):
        self.Image = cv2.imread(flname)
        self.gambar = str(flname)
        self.displayImage(1)
        self.Image2 = self.Image

    def fungsi(self):
        flname,filter = QFileDialog.getOpenFileName(self,'Select Image', 'D:\\Pycham Project\\Pendeteksi Uang Palsu (2)\\Pendeteksi Uang Palsu\\foto\\', "Image Files(*.*)")
        if flname:
           self.loadImage(flname)
        else:
           print('Invalid Image')

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 10
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)



        self.displayImage(2)



    def ssim(self):
        sample = cv2.imread('100.jpg')
        test = self.Image

        #konversi citra sample ke grayscale
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        #menghitung score ssim diantara 2 citra
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)



        #variable diff berfungsi untuk menyimpan perbedaan antara 2 citra
        #lalu dipresentasikan sebagai data float pada rentang array [0,1]
        #jadi array tersebut harus di ubah ke nilai integer 8 bit yang di masukan pada rentang array [0,255]
        #sebelum dapat di gunakan OpenCV
        diff = (diff * 255).astype("uint8")


        #Treshold perbedaan gambar tersebut
        #lalu mencari daerah kontur untuk 2 citra yang diinput
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        #memasukan sample.shape dan datatype ke dalam mask
        #masukan test.copy ke dalam filled_after
        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()


        #mencari objek dengan contour lalu membuat rectangle dan menggambarkanya pada objek tersebut
        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        #membuat output hasil
        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        #jika skor kurang dari 0.97 persen, maka uang palsu jika lebih dari 0.97 maka uang asli
        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))

    def ssim50(self):
        sample = cv2.imread('50.jpg')
        test = self.Image

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)






        diff = (diff * 255).astype("uint8")



        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()



        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))



    def ssim20(self):
        sample = cv2.imread('20.jpg')
        test = self.Image

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)






        diff = (diff * 255).astype("uint8")



        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()



        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))


    def ssim10(self):
        sample = cv2.imread('10.jpg')
        test = self.Image

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)






        diff = (diff * 255).astype("uint8")



        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()



        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))



    def ssim5(self):
        sample = cv2.imread('5.jpeg')
        test = self.Image

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)






        diff = (diff * 255).astype("uint8")



        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()



        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))


    def ssim2(self):
        sample = cv2.imread('2.jpg')
        test = self.Image

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM Score : ", score)






        diff = (diff * 255).astype("uint8")



        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()



        for c in contours:
            area = cv2.contourArea(c)
            if area > 40 :
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x,y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 266, 0), -1)

        cv2.imshow('sample', sample_gray)
        cv2.imshow('citra pengujian', test)
        cv2.imshow('perbedaan menggunalan negative filter', diff)
        cv2.imshow('mask', mask)
        self.Image = filled_after
        self.displayImage(2)

        self.label_3.setText(str("SSIM Score : ") + str(score))

        if score < 0.97 :
            print('Uang adalah Uang Palsu')
            self.label_4.setText(str('Uang adalah Uang Palsu'))
        else:
            print('Uang adalah Uang Asli')
            self.label_4.setText(str('Uang adalah Uang Asli'))



    def displayImage(self,windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)






app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Aplikasi Pendeteksi Uang Kertas Palsu')
window.show()
sys.exit(app.exec_())