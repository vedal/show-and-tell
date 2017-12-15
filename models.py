from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable

class CNN(nn.Module):
    """Class to build new model including all but last layers"""
    def __init__(self, output_dim=1000):
        super(CNN, self).__init__()
        # TODO: change with resnet152?
        pretrained_model = models.resnet18(pretrained=True)
        self.resnet = Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        # weight init, inspired by tutorial
        self.linear.weight.data.normal_(0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.resnet(x)
        x = Variable(x.data)
        x = x.view(x.size(0), -1) # flatten
        x = self.linear(x)

        return x