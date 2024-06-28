# V-CNN
Versatile CNN (for image classification, can be optimized to provide efficient TinyML solutions : good validation accuracy yet low complexity)

Code is avalailable to run in Google COLAB (or in Kaggle platform) 

<a href="https://colab.research.google.com/github/radu-dogaru/V-CNN/blob/main/v_cnn_support%20(1).ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
Code is support for paper (a reprint is available here https://github.com/radu-dogaru/V-CNN/blob/main/VCNN-paper.pdf ):
R. Dogaru and Ioana Dogaru, "V-CNN: A versatile light CNN structure for image recognition on resources-constrained platforms",  submitted to ISEEE-2023, June 24, 2023. 

V-CNN is a less constrained architecture composed of an arbitrary number of macro-layers, each can be 
assigned any degree of non-linearity. The idea of (non)linear macro-layer is simillar to NL-CNN 
but V-CNN allows arbitrary # of filters per each macro-layer and arbitrary nonlinearity. It also allows to 
add any desired number of hidden dense layers in the output classifier. NL-CNN and XNL-CNN models are now 
special cases of the V-CNN. 

News: (Apr. 25, 2024): A Pytorch implementation was added:  https://github.com/radu-dogaru/V-CNN/blob/main/vcnn_pytorch.py
News: (June  28, 2028): A newer Tiny-ML model namned VRES-CNN was derived (it includes V-CNN as a special case). More detalis here: https://github.com/radu-dogaru/vres-cnn 
