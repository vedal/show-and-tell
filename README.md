# kth-sml-project
Scalable Machine Learning project
Image Captioning

training set: MSCOCO Train2014.
val set: subset of MSCOCO Val2014, 2048 images.

batch_size: 128
vocab_size: 15000
CNN resnet34, pretrained on imagenet
lr: 2e-4
embed_size
    group: 256,512
log_step: 125

Experiments:  
Comparing Basic_RNN, LSTM and GRU    
