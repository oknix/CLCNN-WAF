import torch
import torch.nn as nn
import math

# 2回目のmaxpoolにおけるkernel_sizeの値を計算
def calc_maxpool2_input(l_in, kernel_size=7, stride=1, padding=0, dilation=1):
    x_conv1 = math.floor((l_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    x_max1 = math.floor((x_conv1 + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    x_conv2 = math.floor((x_max1 + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    return x_conv2

# CLCNN TypeA
class CLCNN_A(nn.Module):
    def __init__(self, n_classes=1, input_length=1000, input_dim=128, n_conv_filters=64, n_fc_neurons=64, K=7):
        super(CLCNN_A, self).__init__()
        self.embeddings = nn.Embedding(input_length, input_dim)

        self.conv1 = nn.Sequential(
                nn.Conv1d(input_length, n_conv_filters, kernel_size=K, padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=K, stride=1)
                )

        self.conv2 = nn.Sequential(
                nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=K, padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=calc_maxpool2_input(input_dim, kernel_size=K), stride=1)
                )

        self.fc1 = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(num_features=64),
                nn.Dropout(0.5),
                nn.Linear(n_fc_neurons, n_classes),
                nn.Sigmoid()
                )

        if n_conv_filters == 64 and n_fc_neurons == 64:
            self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        output = self.embeddings(input)
        output = self.conv1(output)
        output = self.conv2(output)

        output = output.view(-1, 64)
        output = self.fc1(output)

        return output

# originalのCharacterLevelCNN
class CharacterLevelCNN(nn.Module):
    def __init__(self, n_classes=14, input_length=1014, input_dim=68, n_conv_filters=256, n_fc_neurons=1024):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

if __name__ == '__main__':
    net = CLCNN_A()
    print(net)
    text_input = torch.LongTensor(5, 1000).random_(0, 10)
    output = net(text_input)
    print(output)
    print(output.size())
