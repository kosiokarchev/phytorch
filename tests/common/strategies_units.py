# This tests using a custom dimension
from itertools import repeat

from hypothesis import strategies as st

from phytorch.constants.constant import Constant
from phytorch.units.unit import Dimension, dimensions, Unit


dimensions = dimensions + (Dimension('א'),)

dimensions_strategy = st.fixed_dictionaries({}, optional=dict(zip(dimensions, repeat(st.one_of(
    st.integers(-3, 3), st.floats(-3, 3, allow_nan=False, allow_infinity=False),
    st.fractions(-3, 3), st.decimals(-3, 3), st.sampled_from(('2/3', '-4/5', '2'))
)))))
values_strategy = st.floats(min_value=1e-6, max_value=10, allow_nan=False, allow_infinity=False, exclude_min=True)
units_strategy = st.tuples(dimensions_strategy, values_strategy).map(lambda args: Unit._make(args[0].items(), value=args[1]))
constants_strategy = units_strategy.map(lambda u: Constant('test', u))

