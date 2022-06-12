import torch
import torch.nn as nn
import torchvision.models as models


sigmoid = torch.nn.Sigmoid()


# Gradient reversal class
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


# Gradient reversal function
def grad_reverse(x):
    return GradReverse.apply(x)

# Swish Module For Meta Network
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


# ResNeXt-101 feature extractor
class ResNext101(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNext101, self).__init__()
        self.enet = models.resnext101_32x8d(pretrained=pretrained)
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out

class MetaNN(nn.Module):
    def __init__(self, n_meta_features = 583, n_meta_dim=[512, 128]):
        super(MetaNN, self).__init__()
        self.meta = nn.Sequential(
            nn.Linear(n_meta_features, n_meta_dim[0]),
            nn.BatchNorm1d(n_meta_dim[0]),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            Swish_Module(),
            )
    
    def forward(self,x_meta):
        meta_out = self.meta(x_meta)
        return meta_out

# Main classification head
class ClassificationHead(nn.Module):
    # Define model elements
    def __init__(self, out_dim, in_ch=2048):
        super(ClassificationHead, self).__init__()
        self.layer = nn.Linear(in_ch, out_dim)
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()
        self.dropout = nn.Dropout(0.5)

    # Forward propagate input
    def forward(self, feat_out, meta_out=None):
        # Feature map passed into fully connected layer to get logits
        if meta_out is not None:
            x = torch.cat((feat_out,meta_out),dim=1)
            x = self.layer(self.dropout(x))
        else:
            x = self.layer(self.dropout(feat_out))  # .squeeze()
        # Returning logits
        return x


# Auxiliary head
class AuxiliaryHead(nn.Module):
    # Define model elements
    def __init__(self, num_aux, in_ch=2048):
        super(AuxiliaryHead, self).__init__()
        # Fully connected layer
        self.layer = nn.Linear(in_ch, num_aux)
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()

    # Forward propagate input
    def forward(self, x_aux):
        # Feature map passed into fully connected layer to get logits
        x_aux = self.layer(x_aux).squeeze()
        # Probabilities output by using sigmoid activation
        px_aux = self.activation(x_aux)
        # Returning logits and probabilities as tuple
        return x_aux, px_aux


# Deeper auxiliary head (added fully connected layer)
class AuxiliaryHead2(nn.Module):
    # Define model elements
    def __init__(self, num_aux, in_ch=2048):
        super(AuxiliaryHead2, self).__init__()
        # Fully connected layer with 2 units
        self.layer = nn.Sequential(
               nn.Linear(in_ch, 128),
               nn.ReLU(),
               nn.Linear(128, num_aux))
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()

    # Forward propagate input
    def forward(self, x_aux):
        # Feature map passed into fully connected layer to get logits
        x_aux = self.layer(x_aux).squeeze()
        # Probabilities output by using sigmoid activation
        px_aux = self.activation(x_aux)
        # Returning logits and probabilities as tuple
        return x_aux, px_aux
