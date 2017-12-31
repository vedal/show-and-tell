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


# Neural Image Captioning
The goal of this project was to reimplement the Neural Image Captioning (NIC) model proposed in the article by Vinyals et al.

This project was carried out as part of the ID2223 "Scalable Machine Learning and Deep Learning" course at [KTH Royal Institute of Technology](http://kth.se).

### Contributors
- Martin Hwasser (github: [hwaxxer](https://github.com/hwaxxer/)) 
- Wojciech Kryściński (github: [muggin](https://github.com/muggin/))
- Amund Vedal (github: [amundv](https://github.com/amundv))

### References
The implemented architecture was based on the following publication:
- ["Show and Tell: A Neural Image Captiong Generator" by Vinyals et al.](https://arxiv.org/abs/1411.4555)

### Datasets
Experiments were conducted using the [Common Objects in Context](http://cocodataset.org/) dataset. The following subsets were used:
- Training: 2014 Contest Train images [83K images/13GB]
- Validation: 2014 Contest Val images [41K images/6GB]
- Test: 2014 Contest Test images [41K images/6GB]

### Experiments
#### Research Question
Bla bla bla

#### Setup
Bla bla bla

### Results
#### GRU/LST/Elman
#  <div>
#  <img align="center" src="/misc/ss1.png" width=405>
#  <img align="center" src="/misc/ss2.png" width=415>
#  </div>
