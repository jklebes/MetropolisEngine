import metropolis_engine
import cmath
import numpy as np

#simplest demo of MetropolisEngine (non-adaptive, real parameters only) base class

# the physics system
class System():
  
  def __init__(self,const):
    self.const= const 

  def calc_system_energy(self,state):
    """to be used directly by MetropolisEngine,
     the energy function must take a list"""
    x=state[0]
    y=state[1]
    return self.const*(x**2 + y**2)

  def calc_system_energy_wrong_arg_format(self,x,y):
    return self.const*(x**2 +  y**2)

  def calc_x_energy(self, state):
    return self.const*state[0]**2

  def calc_y_energy(self, state):
    return self.const*state[1]**2

#initialize a system object with desired constants, or otherwise have a function that returns energy
system = System(const=1)

initial_values=np.array([0.0,0.0])
engine = metropolis_engine.MetropolisEngine(initial_real_params=initial_values, temp=0.1)

# set the energy function of the metropolis engine to a function that returns 
# energy based on a list or np array [real paramters, complex parameters] 
# in a pre-agreed order
engine.set_energy_function(system.calc_system_energy)

#alternatively if we have an energy function in a different format,
# first wrap it to the correct format
def my_energy_function(param_list):
  return system.calc_system_energy_wrong_arg_format(*param_list)

engine.set_energy_function(my_energy_function)

#or

engine.set_energy_function(lambda l: system.calc_system_energy_wrong_arg_format(l[0], l[1]))
 
# run the main simulation by looping over metropolis engine's step
values = initial_values
energy = system.calc_system_energy(initial_values)
for i in range(1000):
  for j in range(10):
    values, energy = engine.step_all(values, energy)
    #print(values, energy)
  #measure running mean, covariance matrix estimate every 10 steps
  engine.measure(values)
  

# examine the results: final value of mean, covariance matrix
print(engine.mean)
print(engine.covariance_matrix)

#the above likely didn't work very well because step size was not adaptive
engine2 = metropolis_engine.AdaptiveMetropolisEngine(initial_real_params=initial_values, temp=0.1)
engine2.set_energy_function(system.calc_system_energy)
 
# run the main simulation by looping over metropolis engine's step
values = initial_values
energy = system.calc_system_energy(initial_values)
for i in range(1000):
  for j in range(10):
    values, energy = engine2.step_all(values, energy)
  #measure running mean, covariance matrix estimate every 10 steps
  engine2.measure(values)
print(engine2.mean)
print(engine2.covariance_matrix)

#still has problems
#try a sequential (Gibbs sampling) simulation with the grouping functionality

engine3 = metropolis_engine.AdaptiveMetropolisEngine(initial_real_params=initial_values, temp=0.1)
engine3.set_energy_function(system.calc_system_energy)
engine3.set_energy_terms({0:system.calc_x_energy, 1:system.calc_y_energy}, {0:[0], 1:[1]})
engine3.set_groups({0:[0], 1:[1]},{0:[0], 1:[1]})

values = initial_values
energy =  dict([])
for key in engine3.calc_energy_terms:
  energy[key] = engine3.calc_energy_terms[key](values)
assert(sum(energy.values()) == system.calc_system_energy(initial_values))
for i in range(1000):
  for j in range(5):
    values, energy = engine3.step_group(0, values, energy)
    values, energy = engine3.step_group(1, values, energy)
  engine3.measure(values)
print(engine3.mean)
print(engine3.covariance_matrix)
