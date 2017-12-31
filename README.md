

# Neural Image Captioning
The goal of this project was to tackle the problem of automatic caption generation for images of real world scenes. The work consisted of reimplementing the Neural Image Captioning (NIC) model proposed by Vinyals et al. and running appropriate experiments to tests its performance.

The project was carried out as part of the ID2223 "Scalable Machine Learning and Deep Learning" course at [KTH Royal Institute of Technology](http://kth.se).

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

### Architecture
The NIC architecture consists of two models, the Encoder and a Decoder. The Encoder, which is a Convolutional Neural Network, is used to create a (semantic) summary of the image in a form of a fixed sized vector. The Decoder, which is a Recurrent Neural Network, is used to generate the caption in natural language based on the summary vector created by the encoder.

<p align="center">
<img src="/report/nic-model.png" width=600>
</p>

### Experiments
#### Goals
The goal of the project was to implement and train a NIC architecture and evaluate its performance. A secondary goal, was to check how the type of a recurrent unit and the size of the embeddining in the Decoder (Language Generator) affects the overall performance of the NIC model.

#### Setup
The Encoder was a `ResNet-34` architecture with pre-trained weights on the `ImageNet` dataset. The output layer of the network was replaced with a new layer with a size definable by the user. All weights, except from the last layer, were frozen during the training procedure.

The Decoder was a single layer recurrent neural network. Three different Recurrent units were tested, `Elman`, `LSTML`, and `GRU`.

Training parameters:
- Number of epochs: 3
- Batch size: 128 (3236 batches per epoch)
- Vocabulary size: 15k
- Learning rate: `1e-3`, with LR decay every 2000 batches

Models were implemented in `Python` using the [PyTorch](http://pytorch.org) library.


#### Evaluation Methods
Experiments were evaluated in a qualitative and quantitative manner. Qualitatitve evluation aimed to assess the coherence of the generated sequences and their relevance given the input image. Quantitative evaluation enabled comparison of trained models with reference models from the authors. The following metrics were used: `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`, `ROGUE-L`, `METEOR`, and `CIDEr`. 

### Results
#### Quantitative
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
    <th>ROGUE-L</th>
    <th>METEOR</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td>Vinyal et al.</td>
    <td>BLEU-1</td>
    <td>BLEU-2</td>
    <td>BLEU-3</td>
    <td>BLEU-4</td>    
    <td>ROGUE-L</td>
    <td>METEOR</td>
    <td>CIDEr</td>
  </tr>
  <tr>
    <td>elman256</td>
    <td>61.6</td>
    <td>42.7</td>
    <td>28.4</td>
    <td>19.0</td>  
    <td>45.1</td>
    <td>18.9</td>
    <td>53.7</td>
  </tr>
  <tr>
    <td>elman512</td>
    <td>62.3</td>
    <td>43.5</td>
    <td>29.5</td>
    <td>20.1</td>    
    <td>45.8</td>
    <td>19.7</td>
    <td>58.7</td>
  </tr>
  <tr>
    <td>gru256</td>
    <td>44.9</td>
    <td>24.9</td>
    <td>13.7</td>
    <td>7.6</td>    
    <td>33.4</td>
    <td>14.4</td>
    <td>27.7</td>
  </tr>
  <tr>
    <td>gru512</td>
    <td>44.4</td>
    <td>24.7</td>
    <td>13.5</td>
    <td>7.5</td>    
    <td>33.0</td>
    <td>14.3</td>
    <td>27.6</td>
  </tr>
  <tr>
    <td>lstm256</td>
    <td>59.5</td>
    <td>40.0</td>
    <td>25.9</td>
    <td>16.7</td>    
    <td>43.4</td>
    <td>17.5</td>
    <td>45.6</td>
  </tr>
  <tr>
    <td>lstm512</td>
    <td>60.4</td>
    <td>41.3</td>
    <td>27.2</td>
    <td>18.0</td>    
    <td>44.6</td>
    <td>18.8</td>
    <td>51.2</td>
  </tr>
</table>

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
