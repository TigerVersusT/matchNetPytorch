import torch
import cv2
import numpy as np
import os
from model import ClassiFilerNet

imgDir = "./test/framePatches"
imgList = []
"""
for i in range(36):
    img = cv2.imread(os.path.join(imgDir, str(i)+".jpg"), 0)

    imgList.append(img)
"""
imgList.append(cv2.imread(os.path.join(imgDir, str(21)+".jpg"), 0))

# target patches
imgList = np.array(imgList).reshape(1, 1, 1, 64, 64)
# template patch
template = cv2.imread("./test/template.png", 0) 
template = cv2.resize(template, (64, 64))
template = torch.from_numpy((template-128)/160).to(torch.float32).reshape(1, 1, 64, 64)
print(template.shape)


# load model
model = ClassiFilerNet("alexNet")
pthFile = "./featureNetpth/model_best.pth.tar"
print("=> loading checkpoint '{}'".format(pthFile))
checkpoint = torch.load(pthFile)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
        .format(pthFile, checkpoint['epoch']))


# evluate
model.eval()

results = []
for img in imgList:
    img = torch.from_numpy((img-128)/160).to(torch.float32)

    out = model((template, img))[0]
    print(out)
    results.append(out[1] > 0.55)

print(results)