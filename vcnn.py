# V-CNN Versatile CNN model -
# includes L-CNN, NL-CNN and XNL-CNN as particular cases.
# The basic unit is the "macro-layer" as in the XNL-CNN but here one can independently choose the
# filter size (fil) and
# nonlinearity nl (0 means "linear" convolution)
# It allows any number of additional dense layers e.g. hid=[] (no hidden dense) or hid =[100, 100] (two additional).
# Copyright Radu and Ioana DOGARU - correspondence: radu.dogaru@upb.ro
# Updates: June 21, 2023; April 29, 2024 
#-------------------------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D  # straturi convolutionale si max-pooling
from tensorflow.keras.optimizers import RMSprop, SGD, Adadelta, Adam, Nadam
#--------------------------  ------------------------------
def create_v_cnn_model(input_shape, num_classes, flat=1, fil=[100,100,100,100], nl=[1,1,0,0], hid=[]):
   
    
    # Note the number of elements in fil list (macrolayers) should be the same in nl list
    # hid can be [] while if the are elements, additional dense layers are added in the output classifier

    csize=3; stri=2; psiz=4; pad='same';
    drop1=0.6  # Best value for CIFAR-100 after tuning in range 0.25 - 0.75 !

    nfilmax=np.shape(np.array(fil))[0]

    model = Sequential()
    # First macrolayer - connected to input  ----------------
    if nfilmax>0:

        layer=0
        if nl[layer]>0:
            model.add(Conv2D(fil[layer], padding=pad, kernel_size=(csize, csize), input_shape=input_shape ) )
            model.add(Activation('relu'))
            for nonlin in range(1,nl[0]):
                model.add(Conv2D(fil[layer], padding=pad, kernel_size=(csize, csize)) )
                model.add(Activation('relu'))

            model.add(Conv2D(fil[0], padding=pad, kernel_size=(csize, csize) ) )
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(psiz, psiz),strides=(stri,stri),padding=pad))
            model.add(Dropout(drop1))

        else:
            model.add(Conv2D(fil[0], padding=pad, kernel_size=(csize, csize), input_shape=input_shape ) )
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(psiz, psiz),strides=(stri,stri),padding=pad))
            model.add(Dropout(drop1))
    # The remaining  macro-layers

        for layer in range(1,nfilmax):
        #------------------ nonlin layers -----------------
            for nonlin in range(nl[layer]):
                model.add(Conv2D(fil[layer], padding=pad, kernel_size=(csize, csize)) )
                model.add(Activation('relu'))

        #----------------- default macrolayer output

            model.add(Conv2D(fil[layer], padding=pad, kernel_size=(csize, csize)) )
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(psiz, psiz),strides=(stri,stri),padding=pad))
            model.add(Dropout(drop1))

        # Exit classifier
        # INPUT TO DENSE LAYER (FLATTEN - more data can overfit / GLOBAL - less data - may be a good choice )
        if flat==1:
            model.add(Flatten())  # alternanta cu GlobalAv ..
        elif flat==0:
            model.add(GlobalAveragePooling2D()) # pare sa fie mai Ok la cifar
    elif nfilmax==0:
        model.add(Flatten(input_shape=input_shape))
    nhid=np.shape(np.array(hid))[0]
    if nhid>0:
        for lay in range(nhid):
            model.add(Dense(hid[lay], activation='relu'))
            #model.add(Dropout(drop1))
    model.add(Dense(num_classes, activation='softmax'))

# END OF MODEL DESCRIPTION
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )
    model.build()
    return model

