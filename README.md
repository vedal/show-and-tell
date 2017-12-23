# kth-sml-project
Scalable Machine Learning project
Image Captioning

training set: MSCOCO Train2014. 
val set: subset of MSCOCO Val2014, 132 images. 

batch_size. 
    group: 32. 
    wojtek: 128. 
vocab_size: 15000. 
CNN, pretrained on imagenet
    group: resnet18
    wojtek: resnet34
learningrate
    group: 0.001
    wojtek: ?
embed_size
    group: 256
    wojtek: ?
log_step
    group: 10
    wojtek: ?


Experiments:
Comparing Basic_RNN, LSTM and GRU
