import metropolisengine as me
import cmath

#simplest demo of MetropolisEngine (non-adaptive, real parameters only) base class

# the physics system
class System():
  
  def __init__(const):
    self.const= const 

  def calc_system_energy(state):
    """to be used directly by MetropolisEngine,
     the energy function must take a list"""
    x=state[0]
    y=state[1]
    return self.const*(x**2 + y**2)

  def calc_system_energy_wrong_arg_format(x,y):
    return self.const*(x**2, y**2)

#initialize a system object with desired constants, or otherwise have a function that returns energy
system = System(const=1)

engine = MetropolisEngine(temp=.1)

# set the energy function of the metropolis engine to a function that returns 
# energy based on a list or np array [real paramters, complex parameters] 
# in a pre-agreed order
engine.set_energy_function(system.calc_system_energy)

#alternatively if we have an energy function in a different format,
# first wrap it to the correct format
def my_energy_function(param_list):
  return system.calc_system_energy_wrong_format(*param_list)

engine.set_energy_function(my_energy_function)

#or

engine.set_energy_function(lambda l: system.calc_system_energy_wrong_format(l[0], l[1]))
 
# run the main simulation by looping over metropolis engine's step
for i in range(1000):
  for j in range(10):
    values, energy = engine.step_all(values, energy)
  #measure running mean, covariance matrix estimate every 10 steps
  me.measure()
  

# examine the results: final value of mean, covariance matrix
print(me.mean)
print(me.covariance)
