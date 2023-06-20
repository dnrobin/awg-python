from .core import *
from . import Waveguide

class Aperture(Waveguide.Waveguide):
	"""
	Represents a waveguide cross section to query normal modes and calculate
	overlap for but coupling.
	"""
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
	def index(self):
		pass
	def groupindex(self):
		pass