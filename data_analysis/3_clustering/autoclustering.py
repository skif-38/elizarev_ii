import torch
import matplotlib.pyplot as plt

def func(w):
    return torch.sin(w[0]) * torch.cos(w[1])
epochs = 1000
lr = 1e-1
results = {}

# Adam
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Adam([w], lr=lr)

for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())

results['Adam'] = (x, y, loss)

# LBFGS
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.LBFGS([w], lr=lr, max_iter=1, line_search_fn='strong_wolfe')

def closure():
    optim.zero_grad()
    l = func(w)
    l.backward()
    return l

for step in range(epochs):
    optim.step(closure)
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(func(w).item())

results['LBFGS'] = (x, y, loss)




# Rprop
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Rprop([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['Rprop'] = (x, y, loss)

# Adagrad
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Adagrad([w], lr=lr)

for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['Adagrad'] = (x, y, loss)

# NAdam
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.NAdam([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['NAdam'] = (x, y, loss)

# Adadelta
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Adadelta([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['Adadelta'] = (x, y, loss)



# Adamax
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Adamax([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['Adamax'] = (x, y, loss)

# SGD
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.SGD([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())

results['SGD'] = (x, y, loss)

# ASGD
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.ASGD([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['ASGD'] = (x, y, loss)



# Adafactor
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.Adafactor([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['Adafactor'] = (x, y, loss)

# RMSprop
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.RMSprop([w], lr=lr)

for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()

    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())

results['RMSprop'] = (x, y, loss)

# RAdam
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.RAdam([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['RAdam'] = (x, y, loss)

# AdamW
w = torch.tensor([10.0, 10.0], requires_grad=True)
x, y, loss = [], [], []
optim = torch.optim.AdamW([w], lr=lr)
for step in range(epochs):
    optim.zero_grad()
    function = func(w)
    function.backward()
    optim.step()
    x.append(w[0].item())
    y.append(w[1].item())
    loss.append(function.item())
results['AdamW'] = (x, y, loss)


fig, ax = plt.subplots(figsize=(10, 6))

for name, (px, py, ploss) in results.items():
    ax.plot(ploss, label=f'{name} (loss: {ploss[-1]:.4f})', alpha=0.8)

ax.set_xlabel('Эпохи')
ax.set_ylabel('loss')
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()