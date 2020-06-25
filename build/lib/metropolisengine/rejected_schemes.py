import cmath
import copy
import math
import numpy as np
import random
import scipy.stats
import metropolisengine



class RealImgAdaptiveMetropolisEngine(metropolisengine.RobbinsMonroAdaptiveMetropolisEngine):
  # best for measuring complex variables 
  # short of complex covariance matrix 
  def __init__(self, initial_field_coeffs, initial_amplitude, covariance_matrix=None,
               sampling_width=.05, temp=0):
    super().__init__(initial_field_coeffs=initial_field_coeffs, initial_amplitude=initial_amplitude,
                     sampling_width=sampling_width, temp=temp, covariance_matrix=covariance_matrix)
    self.param_space_dims = 2 * len(initial_field_coeffs) + 1
    if covariance_matrix is None:
      self.covariance_matrix = .2* np.identity(self.param_space_dims)  
    self.m = self.param_space_dims
    
    #these text descriptions should match lists created in contruct_state, construct_observables_state functions
    self.params_names= ["ampiltude"] # key to interpreting output: mean of parameters, covariance matrix
    self.params_names.extend(["real_c_"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)])
    self.params_names.extend(["img_c_"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)])
    self.observables_names = ["abs_amplitude", "amplitude_squared"] # key to interpreting output: other measured means
    self.observables_names.extend(["abs_c_"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)])

  #update mean, cov, observables mean as in abs-robbinsmonro baseclass
  # differnt set of variables happens because  of different constructstate methods
  #def measure(self, amplitude, field_coeffs):
    #super().measure(amplitude, field_coeffs) #update mean (with this subclass' construct state), covariance matrix
    # also update observables mean

  def construct_state(self, amplitude, field_coeffs):
    """
    override function listing which variables are the basis of parameter space for the purpose of sampling
    here: amplitude, real and imaginary parts of field coefficeints
    """
    state = [amplitude] 
    #state.extend([field_coeffs[key].real for key in self.keys_ordered])
    #state.extend([field_coeffs[key].imag for key in self.keys_ordered])
    state.extend([field_coeff.real for field_coeff in field_coeffs])
    state.extend([field_coeff.imag for field_coeff in field_coeffs])
    return np.array(state)

  def construct_observables_state(self, amplitude, field_coeffs):
    """
    override function constructing list of other values whose mean is to be measured.  here: abs of amplitude,
    abs of field coeffs
    """
    state_observables = [abs(amplitude), amplitude**2]
    #state_observables.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    state_observables.extend([abs(field_coeff) for field_coeff in field_coeffs])
    return np.array(state_observables)    

  #TODO make single field_coeff, field coeffs, amplitude, all draw method for each subclass 
  def draw_field_coeff_from_proposal_distribution(self, field_coeff, index):
    proposed_addition_real = random.gauss(0, self.sampling_width**2 * self.covariance_matrix[index, index])
    proposed_addition_imag = random.gauss(0, self.sampling_width**2 * self.covariance_matrix[index+self.num_field_coeffs, index+self.num_field_coeffs])
    proposed_field_coeff = field_coeff + (proposed_addition_real + 1j* proposed_addition_imag)
    return proposed_field_coeff
  
  
  def draw_all_from_proposal_distribution(self, amplitude, field_coeffs):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    state = self.construct_state(amplitude, field_coeffs)
    proposed_addition = np.random.multivariate_normal(mean = [0]*len(state), cov=self.sampling_width**2*self.covariance_matrix)
    proposed_amplitude = amplitude+proposed_addition[0]*(-1 if amplitude <0 else 1)  # bcause prposed state[0] is proposed step in abs(amplitude)
    proposed_field_coeff_additions_real  = proposed_addition[1:self.num_field_coeffs+1]
    proposed_field_coeff_additions_img = proposed_addition[self.num_field_coeffs+1:]
    proposed_field_coeffs = [field_coeff + (add_real + add_img*1j) for (field_coeff,(add_real, add_img)) in zip(field_coeffs, zip(proposed_field_coeff_additions_real, proposed_field_coeff_additions_img))]
    #print("proposed field coeffs",  np.array(proposed_field_coeffs))
    return proposed_amplitude, np.array(proposed_field_coeffs)

  def draw_field_coeffs_from_proposal_distribution(self, amplitude, field_coeffs):
    state = self.construct_state(amplitude, field_coeffs)
    proposed_addition = np.random.multivariate_normal(np.zeros(2*self.num_field_coeffs+1), cov=self.field_sampling_width**2*self.covariance_matrix)
    proposed_field_coeff_additions_real  = proposed_addition[1:self.num_field_coeffs+1]
    proposed_field_coeff_additions_img = proposed_addition[self.num_field_coeffs+1:]
    proposed_field_coeffs = [field_coeff + (add_real + add_img*1j) for (field_coeff,(add_real, add_img)) in zip(field_coeffs, zip(proposed_field_coeff_additions_real, proposed_field_coeff_additions_img))]
    #print("proposed field coeffs",  np.array(proposed_field_coeffs))
    return np.array(proposed_field_coeffs)

class PhasesAdaptiveMetropolisEngine(metropolisengine.RobbinsMonroAdaptiveMetropolisEngine):
  "not a good way to measure complex variables"
  def __init__(self, initial_field_coeffs, initial_amplitude, covariance_matrix=None,
               sampling_width=.05, temp=0):
    super().__init__(initial_field_coeffs=initial_field_coeffs, initial_amplitude=initial_amplitude,
                     sampling_width=sampling_width, temp=temp, covariance_matrix=covariance_matrix)
    self.param_space_dims = 2* self.num_field_coeffs+1
    if covariance_matrix is None:
      self.covariance_matrix =  np.identity(self.param_space_dims)  
    self.m = self.param_space_dims
    
    #these text descriptions should match lists created in contruct_state, construct_observables_state functions
    self.params_names= ["abs_amplitude"] # key to interpreting output: mean of parameters, covariance matrix
    self.params_names.extend(["abs_c_"+str(i) for i in self.keys_ordered])
    self.params_names.extend(["phase_c_"+str(i) for i in self.keys_ordered])
    #self.observables_names = [] #inherit from base class

  def construct_state(self, amplitude, field_coeffs):
    """
    helper function to construct position in parameter space as np array 
    By default the parameters we track for correlation matrix, co-adjustemnt of step sizes are [abs(amplitude), {abs(field_coeff_i)}] (in this order)
    override in derived classes where different representation of the parameters is tracked.
    """
    state = [abs(amplitude)]
    state.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    state.extend([cmath.phase(field_coeffs[key]) for key in self.keys_ordered])
    return np.array(state)

  """
  #not needed because identical to base class
  def construct_observables_state(self, amplitude, field_coeffs):
    #override function constructing list of other values whose mean is to be measured.  here: none
    return np.array([])    
  """


  def draw_field_coeffs_from_proposal_distribution(self, amplitude, field_coeffs):
    state = self.construct_state(amplitude, field_coeffs)
    proposed_addition = np.random.multivariate_normal([0] * len(state),
                                                      self.field_sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    proposed_field_coeff_amplitude = [original + addition for (original, addition) in
                                      zip(state[1:1+num_field_coeffs], proposed_addition[1:1+num_field_coeffs])]
    proposed_phases = [orginal + addition for (original, addition) in zip(state[1+num_field_coeffs:], proposed_addition[1+num_field_coeffs:])]
    proposed_field_coeffs = dict(
      [(key, cmath.rect(proposed_amplitude, proposed_phase)) for (key, (proposed_amplitude,proposed_phase)) in
       zip(self.keys_ordered, zip(proposed_field_coeff_amplitude, proposed_phases))])
    return proposed_field_coeffs

  def draw_all_from_proposal_distribution(self, amplitude, field_coeffs):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    state = self.construct_state(amplitude, field_coeffs)
    proposed_addition = np.random.multivariate_normal([0] * len(state),
                                                      self.sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    proposed_amplitude = amplitude + (-1 if amplitude < 0 else 1) * proposed_addition[0]
    proposed_field_coeff_amplitude = [original + addition for (original, addition) in
                                      zip(state[1:1+num_field_coeffs], proposed_addition[1:1+num_field_coeffs])]
    proposed_phases = [orginal + addition for (original, addition) in zip(state[1+num_field_coeffs:], proposed_addition[1+num_field_coeffs:])]
    proposed_field_coeffs = dict(
      [(key, cmath.rect(proposed_amplitude, proposed_phase)) for (key, (proposed_amplitude,proposed_phase)) in
       zip(self.keys_ordered, zip(proposed_field_coeff_amplitude, proposed_phases))])
    return proposed_amplitude, proposed_field_coeffs



  def draw_field_coeff_from_proposal_distribution(self, field_coeff, index):
    old_phase = cmath.phase(field_coeff)
    amplitude = abs(field_coeff)
    #TODO - draw from multivariate with 2x2 covariance submatrix
    proposed_addition = random.gauss(0, self.sampling_width**2 * self.covariance_matrix[index, index])
    proposed_phase_addition = random.gauss(0, self.sampling_width**2 * self.covariance_matrix[index+self.num_field_coeffs, index+self.num_field_coeffs])
    #print("phase addition", proposed_phase_addition)
    #print("amplitude_addition", proposed_addition)
    proposed_field_coeff = cmath.rect(amplitude + proposed_addition, old_phase+proposed_phase_addition)
    #print("old_coeff", field_coeff)
    #print("new coeff", proposed_field_coeff)
    return proposed_field_coeff

class RelativePhasesAdaptiveMetropolisEngine(PhasesAdaptiveMetropolisEngine):
  "not a good way to measure complex variables - but at least showed me that something is going on with phases"
  def __init__(self, initial_field_coeffs, initial_amplitude, covariance_matrix=None,
               sampling_width=.05, temp=0):
    super().__init__(initial_field_coeffs=initial_field_coeffs, initial_amplitude=initial_amplitude,
                     sampling_width=sampling_width, temp=temp, covariance_matrix=covariance_matrix)

    
    #these text descriptions should match lists created in contruct_state, construct_observables_state functions
    self.params_names= ["abs amplitude"] # key to interpreting output: mean of parameters, covariance matrix
    self.params_names.extend(["abs_c_"+str(i) for i in self.keys_ordered])
    self.params_names.extend(["relative_phase_c_"+str(i) for i in self.keys_ordered[:self.max_field_coeffs]])
    self.params_names.append("phase_c0")
    self.params_names.extend(["relative_phase_c_"+str(i) for i in self.keys_ordered[self.max_field_coeffs+1:]])
    #self.observables_names = [] #inherit from base class
  
  @staticmethod
  def phase_diff(c1, c2):
    phasediff = cmath.phase(c1)- cmath.phase(c2)
    if phasediff > math.pi:
      phasediff -= 2*math.pi
    elif phasediff <= -math.pi:
      phasediff +=2*math.pi
    return phasediff

  def construct_state(self, amplitude, field_coeffs):
    """
    helper function to construct position in parameter space as np array 
    """
    state = [abs(amplitude)]
    state.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    phase0 = cmath.phase(field_coeffs[0])
    state.extend([self.phase_diff(field_coeffs[key], phase0) for key in self.keys_ordered[:self.max_field_coeffs]]) # relative phases of c_-n... c_1 relative to c_0 
    state.append(phase0) # absolute (relative to real line) phase of c_0
    state.extend([self.phase_diff(field_coeffs[key], phase0) for key in self.keys_ordered[self.max_field_coeffs+1:]]) # relative phases of c_1... c_n relative to c_0 
    return np.array(state)
