``phytorch.units``
==================

    *The difference between mathematics and physics is that mathematics deals
    with numbers, whereas physics deals with reality.*

Since physicists (attempt / pretend to) describe *quantitatively* the real
world, they need to associate additional information to the numbers that they
manipulate, turning them into quantities. This information comes in the form of
*units*, which encode, alongside an agreed-upon scale, e.g. how long measurement
sticks are and how fast clocks run, the *dimensionality* of the measurement,
i.e. what sort of phenomenon is described.

Some mathematical operations, like addition, are only allowed for quantities of
the same dimension (i.e. describing the same *things*), whereas others, like
multiplication, can mix and match dimensions, resulting in new quantities.
Finally, the majority of mathematical functions: think $\sin$, $\log$, etc.,
only make sense for pure numbers, which do not directly reference the real world.

.. automodule:: phytorch.units
   :members: Dimension, Unit
   :undoc-members:
