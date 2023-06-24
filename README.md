# V-CNN
Versatile CNN (for image classification, can be optimized to efficient TinyML solutions : good validation accuracy yet low complexity 

Cod in support for paper:
R. Dogaru and Ioana Dogaru, "V-CNN: A versatile light CNN structure for image recognition on resources-constrained platforms",  submitted to ISEEE-2023, June 24, 2023. 

V-CNN is a less constrained architecture composed of an arbitrary number of macro-layers, each can be 
assigned any degree of non-linearity. The idea of (non)linear macro-layer is simillar to NL-CNN 
but V-CNN allows arbitrary # of filters per each macro-layer and arbitrary nonlinearity. It also allows to 
add any desired number of hidden dense layers in the output classifier. NL-CNN and XNL-CNN models are now 
special cases of the V-CNN. 
