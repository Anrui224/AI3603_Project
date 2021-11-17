import torch
import torch.nn as nn
import cv2
import math
import numpy as np
from image_pool import ImagePool

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1),
        )
    def forward(self, input):
        out = self.model(input)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1),
            nn.Flatten(start_dim=2, end_dim= -1),
            nn.Linear(in_features=250000, out_features=1),
            nn.Sigmoid()
        )
    def forward(self, out):
        score = self.model(out)
        return score

class Cycle_GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G1 = Generator()
        self.D1 = Discriminator()
        self.G2 = Generator()
        self.D2 = Discriminator()
        self.pool1 = ImagePool(500)
        self.pool2 = ImagePool(500)

    def forward(self, input1, input2):
        gout = self.G1(input1)
        gin = self.G2(input2)
        outscore = self.D1(gout)
        inscore = self.D2(gin)
        return outscore, inscore
        
    
    def Loss_Adv(G, D, X, Y, size_X, size_Y):
        loss = math.log(D(Y))/size_Y + math.log(1-D(G(X)))/size_X
        return loss
    
    def Loss_Cyc(G1, G2, X, size_X):
        loss = np.linalg.norm(G2(G1(X))-X)/size_X
        return loss

    def backward():
        loss = 0
        loss 



    




if __name__=='__main__':
    img = cv2.imread("C:/Users/Anrui/Desktop/AI_Project/dataset/from_content/banboo/Baidu_0003.jpeg")
    print(img)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).unsqueeze(0).float()
    print(img.shape)
    model = Generator()
    out = model(img)
    out = out.squeeze(0).permute(1,2,0)
    out = out.detach().numpy()
    cv2.imwrite("C:/Users/Anrui/Desktop/AI_Project/test.jpg",out)
