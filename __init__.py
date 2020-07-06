from pkg_resources import get_distribution

from .mestats import *
from .statistics import *
from .metropolis_engine import *
__version__ = get_distribution('metropolisengine').version

