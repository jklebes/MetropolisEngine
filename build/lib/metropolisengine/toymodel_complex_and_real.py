import metropolis_engine as me
import cmath

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

#initialize a system object with desired constants, or otherwise have a function that returns energy
system = System(k=1, alpha=-1, beta=.5)

# now set up a metropolisengine with the desired degree of adaptiveness
# MetropolisEngine  - non-adaptive and real only base class
# AdaptiveMetropolisEngine - adjusts stepsize as a whole
# RobbinsMonroAdaptiveMetropolisEngine - adjusts stepsize individially


engine = me.ComplexAdaptiveMetropolisEngine(initial_real_params = [1.0, 1.0], initial_complex_params = [0+0j],  temp=.1)
engine.set_energy_function(lambda l: system.calc_system_energy(*l))
values = [1.0, 1.0, 0+0j]
energy = system.calc_system_energy(*values)

for i in range(1000):
  for j in range(10):
    values, energy = engine.step_all(values, energy)
  engine.measure(values)

print("mean",engine.mean)
print("cov", engine.covariance_matrix)
