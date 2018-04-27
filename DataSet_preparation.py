import sys
import os
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import zipfile
import io
import cv2

#extract imgages from zip (HRI_GestureDataset_Tsironi.zip)
def extract(filename):
    pwd = 'C:\\Study\\Sem 3\\ChrisTseng\\GRIT_DATASET'
    z = zipfile.ZipFile(filename)
    for f in z.namelist():
        os.chdir(pwd)
        # get directory name from file
        dirname = os.path.splitext(f)[0]
        # create new directory
        os.mkdir(dirname)
        os.chdir(dirname)
        # read inner zip file into bytes buffer
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)
        print(zip_file.namelist())
        for i in zip_file.namelist():
            sam_dir = os.path.splitext(i)[0]
            #os.mkdir(sam_dir)
            #os.chdir(sam_dir)
            print(i)
            z_cont = io.BytesIO(zip_file.read(i))
            sample = zipfile.ZipFile(z_cont)
            sample.extractall()

#
# pwd = 'C:\\Study\\Sem 3\\ChrisTseng\\GRIT_DATASET'
# os.chdir(pwd)
# extract('HRI_GestureDataset_Tsironi.zip')


#Resize the image to 64x48 and return as a numpy array
def getarray(img):
    a = np.asarray(Image.open(img).resize((64,48)))
    #imshow(a)
    return a

#Read images from the given path 
#Resize imgs to 64x48 in grayscale
#return list of numpy array
def getimages(path):
    l = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
#             print(filename)
            file = os.path.join(path,filename)
            l.append(cv2.resize(cv2.imread(file, 0), (64, 48)))
    return np.array(l)

#Given 3 consecutive images 
#Compute temporal differencea and return differential image
def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

# def getDiffrentialImage(img1, img2, img3):
#     #print(img1, img2, img3)
#     return (img2 - img1) ^ (img3 - img2)

#given path to video frames for one action sample
#return differential images for those frames
def getdiffDir(path):
    imgs = getimages(path)
    diff = []
    for i, img in enumerate(imgs[2:-1]):
        im = diffImg(imgs[i-1], img, imgs[i+1])
        #imshow(im)
        diff.append(im)
    return diff



#Given path to extracted Grit 
#Conpute differential images and save them
def getImagedirs(path, savePath):
    for d in os.listdir(path): #GRIT_DATASET
        p = os.path.join(path,d)
        if os.path.isdir(p): #abort
#             os.chdir(p)
            print(d)
            for sampledir in os.listdir(p): # 1_0
                print(sampledir)
                sdir = os.path.join(p,sampledir)
                diff = getdiffDir(sdir) # get diff images for current sample dir
                i = 1
                # break
                diffPath = ""
                for imgArr in diff:
                    # img = cv2.fromarray(imgArr)
                    # img = Image.fromarray(imgArr, mode='L')
                    diffPath = os.path.join(savePath,d,sampledir,str(i)+".jpg")
                    os.makedirs(os.path.dirname(diffPath), exist_ok=True)
                    # img.save(diffPath, 'jpeg')
                    cv2.imwrite(diffPath, imgArr)
                    print('Saved ', diffPath)
                    i = i+1
#             break



os.chdir('C:\\Study\\Sem 3\\ChrisTseng\\GRIT_DATASET\\Images')
cwd = os.getcwd()

saveDir = os.path.join('C:\\Study\\Sem 3\\ChrisTseng\\gesture','Save1')
#saveDir

getImagedirs(cwd, saveDir)