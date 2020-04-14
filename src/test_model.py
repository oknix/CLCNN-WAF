import torch
import torch.nn as nn

INPUT_FEATURES = 2
LAYER1_NEURONS = 3
LAYER2_NEURONS = 3
OUTPUT_NEURONS = 1

activation1 = torch.nn.Tanh()
activation2 = torch.nn.Tanh()
acti_out = torch.nn.Tanh()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(INPUT_FEATURES, LAYER1_NEURONS)
        self.layer2 = nn.Linear(LAYER1_NEURONS, LAYER2_NEURONS)
        self.layer_out = nn.Linear(LAYER2_NEURONS, OUTPUT_NEURONS)

    def forward(self, x):
        x = activation1(self.layer1(x))
        x = activation2(self.layer2(x))
        x = acti_out(self.layer_out(x))

        return x

model = NeuralNetwork()
print(model)

weight_array = nn.Parameter(torch.tensor([[0.6, -0.2]]))


bias_array = nn.Parameter(torch.tensor([0.8]))

model.layer1.weight = weight_array
model.layer1.bias = bias_array

params = model.state_dict()
print(params)


X_data = torch.tensor([[1.0, 2.0]])
print(X_data)

y_pred = model(X_data)
print(y_pred)

x = torch.tensor(1.0, requires_grad=True)

y = x ** 2
print(y)

y.backward()

g = x.grad
print(g)

model.layer1.weight.grad = None
model.layer1.bias.grad = None

X_data = torch.tensor([[1.0, 2.0]])
y_pred = model(X_data)
y_true = torch.tensor([[1.0]])

criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)
loss.backward()

print(model.layer1.weight.grad)
print(model.layer1.bias.grad)

def train_step(train_X, train_y):
    model.train()

    pred_y = model(train_X)

    optimizer.zero_grad()
    loss = criterion(pred_y, train_y)
    loss.backward()

    optimizer.step()

    with torch.no_grad():
        discr_y = discretize(pred_y)
        acc = (discr_y == train_y).sum()

    return (loss.item(), acc.item())

def valid_step(valid_X, valid_y):
    model.eval()

    pred_y = model(valid_X)

    loss = criterion(pred_y, valid_y)

    with torch.no_grad():
        discr_y = discretize(pred_y)
        acc = (discr_y == valid_y).sum()

    return (loss.item(), acc.item())

def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.0)

model.apply(init_parameters)

EPOCHS = 100

avg_loss = 0.0
avg_acc = 0.0
avg_val_loss = 0.0
avg_val_acc = 0.0

train_history = []
valid_history = []

for epoch in range(EPOCHS):
    total_loss = 0.0
    total_acc = 0.0
    total_val_loss = 0.0
    total_val_acc = 0.0
    total_train = 0
    total_valid = 0

    for train_X, train_y in loader_train:
        loss, acc = train_step(train_X, train_y)

        total_loss += loss
        total_acc += acc
        total_train += len(train_y)

    for valid_X, valid_y in loader_valid:
        val_loss, val_acc = valid_step(valid_X, valid_y)

        total_val_loss += val_loss
        total_val_acc += val_acc
        total_valid += len(valid_y)

    n = epoch + 1
    avg_loss = total_loss / n
    avg_acc = total_acc / total_train
    avg_val_loss = total_val_loss / n
    avg_val_acc = total_val_acc / total_valid

    train_history.append(avg_loss)
    valid_history.append(avg_val_loss)

    print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \
          f' loss: {avg_loss:.5f}, acc: {avg_acc:.5f}' \
          f' val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

print('Finished Training')
print(model.state_dict()) 
