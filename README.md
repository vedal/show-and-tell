# Neural Image Captioning
The goal of this project was to tackle the problem of automatic caption generation for images of real world scenes. The work consisted of reimplementing the Neural Image Captioning (NIC) model proposed by Vinyals et al. and running appropriate experiments to test its performance.

The project was carried out as part of the ID2223 "Scalable Machine Learning and Deep Learning" course at [KTH Royal Institute of Technology](http://kth.se).

## To run

#### Install pip packages and cocoapi:
```
pip install -r requirements.txt
git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI/; make install; cd ../..
```

#### Install PyTorch for python 3.5 with CUDA 8.0 (check pytorch.org for other options):

`pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl`

#### Fetch the data (also builds a vocabulary):

`python fetch_data.py # (to also download test set, run python fetch_data --test)`

#### Start training with default arguments (check train.py for more arguments)

`python train.py`

#### Evaluate

`python train.py --sample --checkpoint_file <your-checkpoint-file>`

### Contributors
- Martin Hwasser (github: [hwaxxer](https://github.com/hwaxxer/)) 
- Wojciech Kryściński (github: [muggin](https://github.com/muggin/))
- Amund Vedal (github: [amundv](https://github.com/amundv))


### References
The implemented architecture was based on the following publication:
- ["Show and Tell: A Neural Image Captiong Generator" by Vinyals et al.](https://arxiv.org/abs/1411.4555) [3]


### Datasets
Experiments were conducted using the [Common Objects in Context](http://cocodataset.org/) dataset. The following subsets were used:
- Training: 2014 Contest Train images [83K images/13GB]
- Validation: 2014 Contest Val images [41K images/6GB]
- Test: 2014 Contest Test images [41K images/6GB]


### Architecture
The NIC architecture consists of two models, the Encoder and a Decoder. The Encoder, which is a Convolutional Neural Network, is used to create a (semantic) summary of the image in a form of a fixed sized vector. The Decoder, which is a Recurrent Neural Network, is used to generate the caption in natural language based on the summary vector created by the encoder.

<p align="center">
<img src="/report/nic-model.png" width=600>
</p>

### Experiments
#### Goals
The goal of the project was to implement and train a NIC architecture and evaluate its performance. A secondary goal, was to check how the type of a recurrent unit and the size of the word embeddings in the Decoder (language generator) affects the overall performance of the NIC model.


#### Setup
The Encoder was a `ResNet-34` architecture with pre-trained weights on the `ImageNet` dataset. The output layer of the network was replaced with a new layer with a size definable by the user. All weights, except from the last layer, were frozen during the training procedure.

The Decoder was a single layer recurrent neural network. Three different recurrent units were tested, `Elman`, `GRU`, and `LSTM`. The `Elman` refers to the basic rnn architecture.

Training parameters:
- Number of epochs: `3`
- Batch size: `128` (3236 batches per epoch)
- Vocabulary size: `15,000` most popular words
- Embedding size: `512` (image summary vector, word embeddings)
- RNN hidden state size: `512` and `1024`
- Learning rate: `1e-3`, with LR decay every 2000 batches

Models were implemented in `Python` using the [PyTorch](http://pytorch.org) library. Models were trained either locally or on rented AWS instances (both using GPUs).


#### Evaluation Methods
Experiments were evaluated in a qualitative and quantitative manner. The qualitative evaluation assessed the coherence of the generated sequences and their relevance given the input image, and was done by us manually. The quantitative evaluation enabled comparison of trained models with reference models from the authors. The following metrics were used: `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`, `ROGUE-L`, `METEOR`, and `CIDEr`.

### Results

#### Training Progress
<!-- **Per-batch loss, training set** -->
<div>
  <img align="center" src="/report/plot_train_512.png" width=400>
  <img align="center" src="/report/plot_train_1024.png" width=400>
</div>

<!-- **Average loss over batches, validation set** -->
<div>
  <img align="center" src="/report/plot_val_512.png" width=400>
  <img align="center" src="/report/plot_val_1024.png" width=400>
</div>

#### Quantitative
Qualitative results are presented on the Validation and Test sets. Results obtained with the reimplemented model are compared with the results obtained by the authors of the article.

<table>
  <tr>
    <th colspan="8">Validation Data</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>BLEU-1</th>
    <th>BLEU-2</th>
    <th>BLEU-3</th>
    <th>BLEU-4</th>  
    <th>METEOR</th>
    <th>ROGUE-L</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td>Vinyals et al. (4k subset)</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>27.7</td> 
    <td>23.7</td>
    <td>N/A</td>
    <td>85.5</td>
  </tr>
  <tr>
    <td>elman_512</td>
    <td>62.5</td>
    <td>43.2</td>
    <td>29.1</td>
    <td>19.8</td>  
    <td>19.5</td>
    <td>45.6</td>
    <td>57.7</td>
  </tr>
  <tr>
    <td>elman_1024</td>
    <td>61.9</td>
    <td>42.9</td>
    <td>28.8</td>
    <td>19.6</td>  
    <td>19.9</td>
    <td>45.9</td>
    <td>58.7</td>
  </tr>
  <tr>
    <td>gru_512</td>
    <td>63.9</td>
    <td>44.9</td>
    <td>30.5</td>
    <td>20.8</td>    
    <td>20.4</td>
    <td>46.6</td>
    <td>62.9</td>
  </tr>
  <tr>
    <td>gru_1024</td>
    <td><b>64.0</b></td>
    <td><b>45.3</b></td>
    <td><b>31.2</b></td>
    <td><b>21.5</b></td>    
    <td><b>21.1</b></td>
    <td><b>47.1</b></td>
    <td><b>66.1</b></td>
  </tr>
  <tr>
    <td>lstm_512</td>
    <td>62.9</td>
    <td>44.3</td>
    <td>29.8</td>
    <td>20.3</td>    
    <td>19.9</td>
    <td>46.1</td>
    <td>60.2</td>
  </tr>
  <tr>
    <td>lstm_1024</td>
    <td>63.4</td>
    <td>45.0</td>
    <td>31.0</td>
    <td>21.4</td>    
    <td>20.8</td>
    <td><b>47.1</b></td>
    <td>64.4</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="16">Test Data</th>
  </tr>
  <tr>
    <th><sub>Model</sub></th>
    <th colspan="2"><sub>BLEU-1</sub></th>
    <th colspan="2"><sub>BLEU-2</sub></th>
    <th colspan="2"><sub>BLEU-3</sub></th>
    <th colspan="2"><sub>BLEU-4</sub></th>  
    <th colspan="2"><sub>METEOR</sub></th>
    <th colspan="2"><sub>ROGUE-L</sub></th>
    <th colspan="2"><sub>CIDEr</sub></th>
  </tr>
  <tr>
    <th></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
    <th><sub>c5</sub></th>
    <th><sub>c40</sub></th>
  </tr>
  <tr>
    <td><sub>Vinyals et al.</sub></td>
    <td><sub>71.3</sub></td>
    <td><sub>89.5</sub></td>
    <td><sub>54.2</sub></td>
    <td><sub>80.2</sub></td>
    <td><sub>40.7</sub></td>
    <td><sub>69.4</sub></td>
    <td><sub>30.9</sub></td>
    <td><sub>58.7</sub></td>
    <td><sub>25.4</sub></td>
    <td><sub>34.6</sub></td>
    <td><sub>53.0</sub></td>
    <td><sub>68.2</sub></td>
    <td><sub>94.3</sub></td>
    <td><sub>94.6</sub></td>
  </tr>
  <tr>
    <td><sub>elman_1024</sub></td>
    <td><sub>61.8</sub></td>
    <td><sub>79.9</sub></td>
    <td><sub>42.8</sub></td>
    <td><sub>66.2</sub></td>
    <td><sub>28.7</sub></td>
    <td><sub>51.9</sub></td>
    <td><sub>19.5</sub></td>
    <td><sub>39.8</sub></td>
    <td><sub>19.9</sub></td>
    <td><sub>26.7</sub></td>
    <td><sub>45.7</sub></td>
    <td><sub>58.4</sub></td>
    <td><sub>58.0</sub></td>
    <td><sub>60.0</sub></td>
  </tr>
  <tr>
    <td><sub>gru_1024</sub></td>
    <td><sub>63.8</sub></td>
    <td><sub>81.2</sub></td>
    <td><sub>45.0</sub></td>
    <td><sub>68.1</sub></td>
    <td><sub>30.1</sub></td>
    <td><sub>54.4</sub></td>
    <td><sub>21.3</sub></td>
    <td><sub>42.5</sub></td>
    <td><sub>21.0</sub></td>
    <td><sub>27.8</sub></td>
    <td><sub>47.0</sub></td>
    <td><sub>59.5</sub></td>
    <td><sub>65.4</sub></td>
    <td><sub>66.4</sub></td>
  </tr>
  <tr>
    <td><sub>lstm_1024</sub></td>
    <td><sub>63.3</sub></td>
    <td><sub>81.0</sub></td>
    <td><sub>44.8</sub></td>
    <td><sub>67.9</sub></td>
    <td><sub>30.7</sub></td>
    <td><sub>54.0</sub></td>
    <td><sub>21.1</sub></td>
    <td><sub>42.0</sub></td>
    <td><sub>20.7</sub></td>
    <td><sub>27.4</sub></td>
    <td><sub>46.9</sub></td>
    <td><sub>59.2</sub></td>
    <td><sub>63.7</sub></td>
    <td><sub>64.8</sub></td>
  </tr>
</table>
  
_Note: The "MSCOCO c5" dataset contains five reference captions for every image in the MS COCO training, validation and testing datasets. "MSCOCO c40" contains 40 reference sentences for a randomly chosen 5,000 images from the MS COCO testing dataset[2]_.  

_Note2: We assume the score "Vinyals et al." is the top score of the first author of our main reference paper [3]. It comes from the MSCOCO leaderboard at www.codalab.org, where we evaluated our scores. We decided to map the scores from the range [0,1] displayed on the website to [0,100] to keep consistency with previous table and Vinyals et al. Our results for `GRU` and `LSTM` are on place 84 and 85 of the codalab leader board, respectively._



#### Qualitative
**Captions without errors** (left-to-right: Elman, GRU, LSTM)
<div>
  <img align="center" src="/report/example1-elman.png" width=275>
  <img align="center" src="/report/example1-gru.png" width=275>
  <img align="center" src="/report/example1-lstm.png" width=275>
</div>


**Captions with minor errors** (left-to-right: Elman, GRU, LSTM)
<div>
  <img align="center" src="/report/example2-elman.png" width=275>
  <img align="center" src="/report/example2-gru.png" width=275>
  <img align="center" src="/report/example2-lstm.png" width=275>
</div>


**Captions somewhat related to images** (left-to-right: Elman, GRU, LSTM)
<div>
  <img align="center" src="/report/example3-elman.png" width=275>
  <img align="center" src="/report/example3-gru.png" width=275>
  <img align="center" src="/report/example3-lstm.png" width=275>
</div>


**Captions unrelated to image** (left-to-right: Elman, GRU, LSTM)
<div>
  <img align="center" src="/report/example4-elman.png" width=275>
  <img align="center" src="/report/example4-gru.png" width=275>
  <img align="center" src="/report/example4-lstm.png" width=275>
</div>


### Discussion
Studying the results of our experiments, we noted how increasing the number of hidden units describing the RNN state resulted in improved performance across all models, which matched our expectations. However, it was interesting to see the `GRU` cell outperform `LSTM` in both experiments. A possible explanation of this is that for generating relatively short sequences (most captions had up to 20 words) the architecture of the `LSTM` cell might be overly complex. The sequences might also be too short for the `LSTM` to shine. Since the `LSTM` has more trainable parameters when compares to `GRU` it would be interesting to see if extending the training procedure in the case of `LSTM`-based networks allows them to obtain the same or better performance than `GRU`-based networks.


### References: 
[1] Chung et al. (2014) Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. arXiv:1412.3555.  
[2] X. Chen et al. (2015) Microsoft COCO Captions: Data Collection and Evaluation Server. arXiv:1504.00325.  
[3] O. Vinyals et al. (2014) Show and Tell: A Neural Image Captiong Generator, arXiv:1411.4555
