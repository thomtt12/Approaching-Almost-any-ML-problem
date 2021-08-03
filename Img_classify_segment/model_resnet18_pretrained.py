
# model.py 
# using ResNet18 (Residual Neural Network 18, 34, 50 , 101, 152 layer) with pretrained weights on ImageNet dataset
#ResNet consist of residual blocks that tranfer the knowledge from one layer to futher layers by skipping some layers in between
#These kinds of connections of layers are known as skip-connection since skipping one or more layers
#Skip-connection help with the vanishing gradient issue by propagating the gradients to further layers
#Allow to train very large convolutional neural networks without loss of performance



import torch.nn as nn
import pretrainedmodels
def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet18"](
        pretrained='imagenet'
        )
    else:
        model = pretrainedmodels.__dict__["resnet18"](
        pretrained=None
        )
        # print the model here to know whats going on.
        model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1),
        )
        return model