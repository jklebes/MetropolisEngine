import metropolisengine
import cmath
import numpy as np

#simplest demo of MetropolisEngine (non-adaptive, real parameters only) base class

# the physics system
class System():
  
  def __init__(self,const=1):
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
engine = metropolis_engine.MetropolisEngine(energy_function = lambda real_params, complex_params : system.calc_system_energy(real_params), initial_real_params=initial_values, temp=0.1)

 
# run the main simulation by looping over metropolis engine's step
values = initial_values
engine.energy = {"total":system.calc_system_energy(initial_values)}
for i in range(1000):
  for j in range(10):
    engine.step_all()  # should be redirected to step_real_group, because no complex parameters were initialized
    #print(engine.real_params, engine.complex_params, engine.energy)
  #measure running mean, covariance matrix estimate every 10 steps
  engine.measure() #it;s not necessary to record the values at every step , rather record at about correlation time
  #print("adjusted stepsize to ", engine.real_group_sampling_width)
  

# examine the results: final value of mean, covariance matrix
print("mean", engine.real_mean, engine.complex_mean)
print("cov", engine.covariance_matrix_real, engine.covariance_matrix_complex)
print(list(zip(engine.observables_names, engine.observables)))
