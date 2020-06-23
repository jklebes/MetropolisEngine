import metropolisengine as me
import cmath

# assume we have a physics system consisting of a uniform complex-valued field on a plane,
# constrained by the potential E = int ( alpha |c|^2 + beta |c|^4 ) = x*y(alpha |c|^2 + beta |c|^4)
# and the size of the plane is limited by the pressure k

# the following object contains its constants alpha, beta and a function to return the energy
class System():
  
  def __init__(k, alpha, beta):
    self.k = k 
    self.alpha = alpha
    self.beta = beta

  def calc_system_energy(x, y, c):
    area_term = self.k *(1-x) + self.k*(1-y)
    field_term = x*y*(self.alpha *c*c.conjugate() + self.beta *c**2*c.conjugate()**2)
    return area_term + field_term

#initialize a system object with desired constants, or otherwise have a function that returns energy
system = System(k=1, alpha=-1, beta=.5)

# now set up a metropolisengine with the desired degree of adaptiveness
# MetropolisEngine  - non-adaptive and real only base class
# AdaptiveMetropolisEngine - adjusts stepsize as a whole
# RobbinsMonroAdaptiveMetropolisEngine - adjusts stepsize individially


engine = ComplexAdaptiveMetropolisEngine(temp=.1)
engine.set_energy_function(system.calc_system_energy)

for i in range(1000):
  values, energy = engine.step_all(values, energy)

print(values, energy)
