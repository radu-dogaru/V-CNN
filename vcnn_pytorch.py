# V-CNN Pytorch model - in support of the paper (to be cited)
# available in https://github.com/radu-dogaru/V-CNN
'''
R. Dogaru and I. Dogaru, "V-CNN: A Versatile Light CNN Structure For Image Recognition 
On Resources Constrained Platforms," 2023 8th International Symposium on 
Electrical and Electronics Engineering (ISEEE), Galati, Romania, 2023, pp. 44-47, 
doi: 10.1109/ISEEE58596.2023.10310339.
'''
   
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class V_CNN(nn.Module):
    def __init__(self, input_shape, num_classes, flat=0, fil=[20,30,40], nl=[2,1,0]):                                             
        super().__init__()
        
        csize=3; stri=2; psiz=4; pad='same'; drop1=0.6   # 
        padm=1; 
        nfilmax=nfilmax=np.array(fil).shape[0]  
        if padm==0:
            im_size = np.array(list(input_shape)[1:3]) // 2 -1  
        elif padm==1:
            im_size = np.array(list(input_shape)[1:3]) // 2 
        
        self.layers = nn.Sequential()   
        
        layer=0
        if nl[layer]>0:
            self.layers.append( nn.Conv2d(list(input_shape)[0], fil[layer], csize, 1, pad)) ,   
            self.layers.append(nn.ReLU())
            for nonlin in range(1,nl[0]):
                self.layers.append( nn.Conv2d(fil[layer], fil[layer], csize, 1, pad)) ,  
                self.layers.append(nn.ReLU())
            
            self.layers.append( nn.Conv2d(fil[layer], fil[layer], csize, 1, pad))
            self.layers.append(nn.BatchNorm2d(fil[layer]))
            self.layers.append(nn.MaxPool2d(kernel_size=psiz, stride=stri, padding=padm))
            self.layers.append(nn.Dropout(drop1))    
        else:
            self.layers.append( nn.Conv2d(list(input_shape)[0], fil[layer], csize, 1, pad))
            self.layers.append(nn.BatchNorm2d(fil[layer]))
            self.layers.append(nn.MaxPool2d(kernel_size=psiz, stride=stri, padding=padm))
            self.layers.append(nn.Dropout(drop1))    
        if padm==0:
            im_size = im_size // 2 -1   
        elif padm==1:
            im_size = im_size // 2 
        
        for layer in range(1,nfilmax):
            if nl[layer]>0:
              self.layers.append( nn.Conv2d(fil[layer-1], fil[layer], csize, 1, pad)) ,   
              self.layers.append(nn.ReLU())
              for nonlin in range(1,nl[layer]):
                  self.layers.append( nn.Conv2d(fil[layer], fil[layer], csize, 1, pad)) ,   
                  self.layers.append(nn.ReLU())
              
              self.layers.append( nn.Conv2d(fil[layer], fil[layer], csize, 1, pad))
              self.layers.append(nn.BatchNorm2d(fil[layer]))
              self.layers.append(nn.MaxPool2d(kernel_size=psiz, stride=stri, padding=padm))
              self.layers.append(nn.Dropout(drop1))    
            else:
          
              self.layers.append( nn.Conv2d( fil[layer-1], fil[layer], csize, 1, pad) )
              self.layers.append(nn.BatchNorm2d(fil[layer]))
              self.layers.append(nn.MaxPool2d(kernel_size=psiz, stride=stri, padding=padm))
              self.layers.append(nn.Dropout(drop1))
            if layer<(nfilmax-1):
              if padm==0:
                im_size = im_size // 2 -1  
              elif padm==1:
                im_size = im_size // 2
              
        if flat==0:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else :
            self.gap = nn.Flatten(1,1)
        
        if flat==0:
          
          dim_iesire=fil[nfilmax-1] 
        elif  flat==1:
          dim_iesire=int(fil[nfilmax-1]*int(im_size[0]*im_size[1])) 
        self.fc = nn.Linear(dim_iesire, int(num_classes))

    def forward(self, x):
        x = self.layers(x)
        # global average pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.sigmoid(x)  
        #x = F.softmax(x, dim=0) 
        return x

#=========== example model =====================

net=V_CNN(input_shape,num_classes, flat=0, fil=[80,120,160,170,200,200], nl=[1,1,0,0,0,0])
from torchsummary import summary
use_cuda = torch.cuda.is_available()
if use_cuda:
    summary(net.cuda(), input_size = input_shape, batch_size = -1)
else:
    summary(net, input_size = input_shape, batch_size = -1)
