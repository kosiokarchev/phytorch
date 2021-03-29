import torch
from pyrofit.utils.interpolate import LinearNDGridInterpolator
from phytorch.units.unit import Unit
from phytorch.units.quantity.torchquantity import TorchQuantity


Quantity = TorchQuantity
from phytorch.units.si import m, s, kg

q = Quantity([1, 2, 3, 4, 5, 6, 7, 8, 9], unit=s).reshape((3, 3))
q.requires_grad_(True)

x = Quantity(torch.linspace(0, 1, 101), unit=s)
y = Quantity(((x/(Quantity([1], unit=s)))**2).to(Unit()), unit=kg)
xnew = Quantity((torch.rand(150) / 5), unit=5*s)

ipol = LinearNDGridInterpolator((x,), y)
res = ipol(xnew.unsqueeze(-1))

from matplotlib import pyplot as plt

plt.plot(x.numpy(), y.numpy())
plt.plot(xnew.to(x.unit).numpy(), res.to(y.unit).numpy(), '.')
