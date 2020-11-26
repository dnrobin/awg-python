"""
awg is a package for the design and simulation of Arrayed Waveguide Gratings.
"""

__version__ = "1.0.0"

from .AWG import (
    AWG,
    iw,
    aw,
    ow,
    fpr1,
    fpr2,
)

from .Field import Field
from .Aperture import Aperture
from .Waveguide import Waveguide
from .SimulationOptions import SimulationOptions

from .material import (
    Material,
    dispersion
)