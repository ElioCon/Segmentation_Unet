# Segmentation_Unet
The semantic segmentation project using CityScapes dataset. 

### Submission information: 
- Codalab username: Dmitry
- Email: d.stepanov@student.tue.nl

### Instructions
- In order to run the model, use train.py and the model file model_....py file. Change the model file name to model.py and uncomment the desired model in the train.py. Run train.py to train the model. 
- More convinient method to run the models locally is by running the notebook Models.ipynb. Follow the instructions in the notebook to test all the provided models. 

### Types of models presented
There are multiple types of models: 
- Baseline, a simple U-net model. This model already includes early stopping and dropout layers. 
- U-net with the backbone of ResNet and with addition of the attention layers. This model uses ResidualBlock in train.py and attention mechanism.
- U-net with the backbone of the EfficientNet. This model uses MBConv block at the encoder side.

### Improvements over baseline:
- Xavier weights initialization adjusted to ReLU layers.
- Weights normalization at each layer using batch normalization. 
- Addition of early stopping
- Addition of drop out layers
- Addition of scheduler to adjust the update step during the optimization based on the number of epochs trained. 

### Required libraries:

matplotlib.pyplot
numpy
torchvision
torchvision.transforms.v2
torchvision
argparse
torch.optim.lr_scheduler
torch
gc
os
random
pdb
collections
 matplotlib.colors
math
