from tkinter import *
import easyocr
import pandas as pd
import cv2
img = cv2.imread(r"D:\Saved Pictures\pic 17.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noise =cv2.medianBlur(gray,3)
thresh =cv2.threshold(noise,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
reader = easyocr.Reader(["en"])
result =reader.readtext(img)
df=pd.DataFrame(result)
print(df[1])
