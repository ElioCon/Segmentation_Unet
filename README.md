# Segmentation_Unet
The semantic segmentation project using CityScapes dataset. 

In order to run the model, use train.py and the model file model_....py file. Change the model file name to model.py and uncomment the desired model in the train.py. Run train.py to train the model. More convinient method to run the model locally is the the notebooks with the similar names.
There are multiple types of models: 
- Baseline, a simple U-net model. Files: model_baseline.py and model_baseline.ipynb. 
- Baseline with drop out (model_baseline_dp)
- U-net with the backbone of ResNet. The residual connections of the ResNet are used in order to keep the gradient at the deeper layers. This model uses ResidualBlock in the train.py.
- U-net with the backbone of ResNet and with addition of the attention layers. This model uses ResidualBlock in train.py and attention mechanism.
- U-net with the backbone of the EfficientNet. This model uses 
