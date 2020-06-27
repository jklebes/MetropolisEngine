import metropolisengine
import cmath
import numpy as np

# assume we have a physics system consisting of a uniform complex-valued field on a plane,
# constrained by the potential E = int ( alpha |c|^2 + beta |c|^4 ) = x*y(alpha |c|^2 + beta |c|^4)
# and the size of the plane is limited by the pressure k

# the following object contains its constants alpha, beta and a function to return the energy
class System():

  def __init__(self, k, alpha, beta):
    self.k = k
    self.alpha = alpha
    self.beta = beta

  def calc_system_energy(self, x, y, c):
    area_term = self.k *(1-x)**2 + self.k*(1-y)**2
    field_term = x*y*(self.alpha *c*c.conjugate() + self.beta *c**2*c.conjugate()**2)
    return area_term + field_term
  
  def calc_field_energy(self, x,y, c):
    return x*y*(self.alpha *c*c.conjugate() + self.beta *c**2*c.conjugate()**2)
 
  def calc_area_energy(self, x, y):
    return  self.k *(1-x)**2 + self.k*(1-y)**2

#initialize a system object with desired constants, or otherwise have a function that returns energy
system = System(k=1, alpha=-1, beta=.5)

initial_real_values=np.array([0.0,0.0])
initial_complex_values = np.array([0+0j])
field_fct = lambda r, c : system.calc_field_energy(*r, *c)
area_fct =  lambda real_params,complex_params : system.calc_area_energy(*real_params)
energy_fcts = {"complex": {"field":field_fct}, "real":{"field": field_fct, "area": area_fct}, "all": {"field":field_fct, "area": area_fct}} #which energy terms change when "real" or "complex" groups of parameters change, and how they are to be re-evaluated
engine = metropolis_engine.MetropolisEngine(energy_functions = energy_fcts, initial_real_params=initial_real_values, initial_complex_params = initial_complex_values, temp=0.1)

 
# run the main simulation by looping over metropolis engine's step
#engine.energy = system.calc_system_energy(*initial_real_values, *initial_complex_values)
for i in range(100):
  for j in range(10):
    engine.step_all()
    #print(engine.real_params, engine.complex_params, engine.energy)
  #measure running mean, covariance matrix estimate every 10 steps
  engine.measure() #it;s not necessary to record the values at every step , rather record at about correlation time
  #print("adjusted stepsize to ", engine.real_group_sampling_width, engine.complex_group_sampling_width)
  

# examine the results: final value of mean, covariance matrix
print("mean", engine.real_mean, engine.complex_mean)
print("cov", engine.covariance_matrix_real, engine.covariance_matrix_complex)
print(list(zip(engine.observables_names, engine.observables)))

engine.save_time_series()
print(engine.df)
