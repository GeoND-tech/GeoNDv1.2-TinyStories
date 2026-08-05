### Overview

This repository is meant as a proof of concept for how paraboloid neurons can be used to achieve similar performance with significantly fewer parameters by replacing the entire feed-forward component of a transformer with a single paraboloid neuron layer.

|   Model           | Epochs        | Parameters (Millions) | Training loss | Validation loss |
| ----------------- | -------------        | -------- | -------- | -------- |
| ```NanoGPT```, baseline       | 3       | 4.484M | 0.4498577415943146 | 0.4255075752735138 |
| ```NanoGPT```, reduced feed forward       | 3       | 2.908M | 0.47032883763313293 | 0.4420813024044037 |
| **```NanoGPT```, paraboloid feed forward**        | **3**       | **2.908M** | **0.45499327778816223** | **0.4283764958381653** |



# Using paraboloid neurons to train NanoGPT models on TinyStories with PyTorch

Paraboloid neuron demonstration of the [GeoND Library](https://geond.tech) for [PyTorch](http://pytorch.org/) on the TinyStories dataset. If you are interested in trying out the library on your own datasets, please refer to the ["How to use"](https://geond.tech/geond-docs/) section of the documentation. This repository uses Version 1.2 of the GeoND Library. You can find download instructions here: [https://geond.tech/download/](https://geond.tech/download/). Adapted from [https://github.com/broskicodes/slms](https://github.com/broskicodes/slms).

## Requirements
- Linux only.
- Python 3.9+, use of a virtual environment recommended.
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```
- (Optional) Download the pre-trained models by running:
```
wget -i models.txt -P models
```

## IMPORTANT
Including any layer with paraboloid neurons requires a specialized optimizer. As of version 1.1, the GeoND Library includes an adaptation of Adam and AdamW:
```
optimizer = gpt.GeoNDAdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999, 0.999))
```

## Models
- ### NanoGPT
Our baseline GPT model.
#### Generate a story
Download the pretrained model and run:
```
python small_lms_test.py
```
#### Training from scratch
Run:
```
python small_lms_train.py
```

- ### NanoGPT_paraboloid
A GPT model with a layer of paraboloid neurons replacing the first layer of the FeedForward component. We used 1/4 of the neurons of the original layer, in order to come up with a smaller model.

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the original component:
```
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_embed),
      nn.Dropout(dropout)
    )
```
By replacing the first hidden layer with a ```Paraboloid``` layer:
```
    self.pb = gpt.Paraboloid(n_embed, n_hidden, lr_factor = 10., input_factor = 0.1, wd_factor = 0.01)
    self.rl = nn.ReLU()
    self.ln = nn.Linear(n_hidden, n_embed)
    self.dr = nn.Dropout(dropout)
```
Since ```Paraboloid``` does not handle 3D tensor inputs internally, we also have to update the forward function:
```
  def forward(self, x):
   
   x_shape = x.shape
   x_reshaped = x.reshape(-1, x_shape[2])
   out = self.pb(x_reshaped)
   out = out.reshape(x_shape[0],x_shape[1],self.pb.output_features)
   out = self.rl(out)
   out = self.ln(out)
   out = self.dr(out)
```
We also use 1/4 of the neurons here:
```
    #self.ffwd = FeedForward(n_embed, n_embed*4, dropout)
    self.ffwd = FeedForward(n_embed, n_embed, dropout)
```
Note that the smaller model requires more epochs to reach similar loss function values, namely 10 instead of 3.

#### Generate a story
Download the pretrained model and run:
```
python small_pblms_test.py
```
#### Training from scratch
Run:
```
python small_pblms_train.py
```

## Evaluation of pretrained models


## References
- Original repository: [https://github.com/broskicodes/slms](https://github.com/broskicodes/slms)
- GeoND Library documentation: [https://geond.tech/geond-docs/](https://geond.tech/geond-docs/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)
