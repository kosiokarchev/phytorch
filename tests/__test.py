import torch
from pyrofit.utils.interpolate import Linear1dInterpolator
from phytorch.units.Unit import Unit
from phytorch.quantities.torchquantity import TorchQuantity as Quantity
from phytorch.units.si import m, s, kg

# noinspection PyUnresolvedReferences
from clipppy.patches import torch_numpy


q = Quantity([1, 2, 3, 4, 5, 6, 7, 8, 9], unit=s).reshape((3, 3))
q.requires_grad_(True)

x = Quantity(torch.linspace(0, 1, 11), unit=s).requires_grad_(True)
y = Quantity(((x/(Quantity([1], unit=s)))**2).to(Unit()), unit=kg).requires_grad_(True)
xnew = Quantity((torch.rand(150) / 5), unit=5*s)

ipol = Linear1dInterpolator(x, y)
ynew = ipol(xnew)

from matplotlib import pyplot as plt

plt.plot(x.numpy(), y.numpy())
plt.plot(xnew.to(x.unit).numpy(), ynew.to(y.unit).numpy(), '.')

# Gradient is not what you expect (well, partly).....
# x.retain_grad(), y.retain_grad()
# ynew.sum().backward()
