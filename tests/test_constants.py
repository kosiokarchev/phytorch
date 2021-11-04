from hypothesis import given
from pytest import mark

from phytorch import constants as consts
from phytorch.constants.codata import CODATA
from phytorch.constants.constant import Constant
from phytorch.units.si import metre
from phytorch.units.unit import Unit

from tests.common.strategies_units import constants_strategy


def test_new_constant():
    smoot = Constant('smoot', (u := 1.7 * metre), 'Smoot\'s height')
    assert smoot.name == 'smoot'
    assert smoot == u
    assert smoot.description == 'Smoot\'s height'


@given(constants_strategy)
def test_constant_arithmetic(c: Constant):
    assert isinstance(c, Unit)
    assert c.to(Unit(c)) == c.value
    assert (u1 := 2 * c).value == 2*c.value and u1.dimension == c.dimension
    assert (u2 := c**2).value == c.value**2 and u2.dimension == (Unit(c)**2).dimension


def test_module_getattr():
    assert isinstance(consts.codata2014, CODATA)
    assert isinstance(consts.codata2018, CODATA)
    assert isinstance(consts.default, CODATA)
    assert consts.default is consts.codata2018


@mark.parametrize('name', consts.default.__all__ + ('codata2014', 'codata2018', 'default'))
def test_module_dir(name):
    assert name in dir(consts)


@mark.parametrize('name', consts.default.__all__)
def test_default_constants(name):
    assert (c := getattr(consts, name)) is getattr(consts.default, name)
    assert isinstance(c, Constant)
