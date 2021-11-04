from numbers import Real
from typing import Union

from ..units.unit import Unit


class Constant(Unit):
    # necessary because Constant.__init__ has a different signature
    @classmethod
    def _make(cls, iterable, **kwargs):
        return Unit._make(iterable, **kwargs)

    def __init__(self, name, unit: Union[Unit, Real], description=''):
        if not isinstance(unit, Unit):
            unit = Unit(value=unit)
        super().__init__(unit, value=unit.value, name=name)
        self.unit_name = unit.name
        self.description = description

    def __repr__(self):
        return f'<{self.description or type(self).__name__}: {self.name} = {self.unit_name}>'
