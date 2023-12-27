import torch.nn
import torch.nn.functional as F


class EEGNet(torch.nn.Module):
    def __init__(self, activation: str="relu"):
        super(EEGNet, self).__init__()

        self.activation = activation

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), stride=(1, 32), padding=(0, 32), bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
        self.dropout1 = torch.nn.Dropout(p=0.25, inplace=False)

        # Depthwise convolution
        self.depthwiseConv1 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(16, 1), stride=(1, 1), padding=(0, 8), groups=4, bias=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1), padding=0, dilation=1, ceil_mode=False)
        self.dropout2 = torch.nn.Dropout(p=0.25, inplace=False)    
    

        # Linear layer
        self.linear1 = torch.nn.Linear(in_features=4480, out_features=2, bias=True)

    
    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        # Apply the activation function
        if self.activation == "relu":
            x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Depthwise convolution
        x = self.depthwiseConv1(x)
        # Apply the activation function
        if self.activation == "relu":
            x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # FCC layer
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)

        return x

class testEEGNet(torch.nn.Module):
    def __init__(self, activation: str="relu"):
        super(testEEGNet, self).__init__()

        self.activation = activation

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), stride=(1, 32), padding=(0, 32), bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
        self.dropout1 = torch.nn.Dropout(p=0.25, inplace=False)

        # Depthwise convolution
        self.depthwiseConv1 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(16, 1), stride=(1, 1), padding=(0, 8), groups=4, bias=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1), padding=0, dilation=1, ceil_mode=False)
        self.dropout2 = torch.nn.Dropout(p=0.25, inplace=False)   

        # 1 point convolution
        self.pointConv1 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)    

        # Linear layer
        self.linear1 = torch.nn.Linear(in_features=280, out_features=2, bias=True)

    
    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        # Apply the activation function
        if self.activation == "relu":
            x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Depthwise convolution
        x = self.depthwiseConv1(x)
        # Apply the activation function
        if self.activation == "relu":
            x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # 1 point convolution
        x = self.pointConv1(x)

        # FCC layer
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)
        return x

class psd_EEGNet(torch.nn.Module):
    def __init__(self, activation: str="relu"):
        super(psd_EEGNet, self).__init__()

class spec_EEGNet(torch.nn.Module):
    def __init__(self, activation: str="relu"):
        super(spec_EEGNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.batchnorm3 = torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool3 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    

    def forward(self, x):
        # Convolutional layer 1
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.avgpool1(x)

        # Convolutional layer 2
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.avgpool2(x)

        # Convolutional layer 3
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.avgpool3(x)

        # FCC layer
        x = x.view(x.shape[0], -1)
        return x
    
if __name__ == "__main__":
    # Create a random input
    input = torch.randn(10, 1, 128, 512)

    # Create the model
    model = testEEGNet()

    # Run the model
    output = model(input)
    print(output.shape)