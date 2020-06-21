import cmath
import copy
import math
import numpy as np
import random
import scipy.stats


class MetropolisEngine():
  """
  base version: proposal distribution is fixed in width and covariance.
  By default the covariance matrix of the multivariate proposal distribution is the dientity matrix, equivalent to drawing from n independent gaussian distribution.
  A different fixed covariance matrix can alo be chosen.
  draws steps from n separate gaussian (or other) distributions.
  Using just this base class with constant (in run time), global (in parameter space) proposal distribution guarantees ergodic sampling.
  Subclass StaticCovarianceAdaptiveMetropolisEngine is recommended minimum level of adaptiveness.
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=None, sampling_width=0.05, temp=0, covariance_matrix=None,
               target_acceptance=.3):
    """
    :param num_field_coeffs: number of field coefficient degrees of freedom.
    :param sampling_width: float, width of proposal distribution.  A factor scaling covariance matrix in multivariate gaussian proposal distribution.
    :param temp: system temperature.  Optional, default 0.
    :param covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of free parameters: in general number of field coefficients + 1.  2*number of field coefficients+1 Where phase/ampltitude or real/imag parts of field coefficients are treated as separate degrees of freedom.  complex entries in subclass ComplexAdaptiveMetropolisEngine.
    """
    self.temp = temp
    assert (self.temp is not None)
    self.num_field_coeffs = len(initial_field_coeffs)
    self.max_field_coeffs = (self.num_field_coeffs-1)//2
    #self.keys_ordered = list(range(-self.max_field_coeffs, self.max_field_coeffs + 1))
    self.param_space_dims = self.num_field_coeffs + 1  # in more basic case perturbation amplitude, magnitudes only of ci.  subclasses may consider both magnitude and phase / both real and img parts of cis and increase this number.
    if covariance_matrix is None:
      self.covariance_matrix = np.identity(
        self.param_space_dims)  # initial default guess needs to be of the right order of magnitude of variances, or covariance matrix doesnt stabilize withing first 200 steps before sigma starts adjusting
    else:
      self.covariance_matrix = covariance_matrix
    #self.accepted_counter = 1
    self.step_counter = 1
    self.measure_step_counter = 1
    self.amplitude_step_counter = 1
    self.field_step_counter = 1
    if isinstance(sampling_width,list):
      self.field_sampling_width, self.amplitude_sampling_width = sampling_width
      self.sampling_width = self.field_sampling_width #arbitrarily use this one when generating a small number to stabilize cov matrix
    else:
      self.sampling_width = sampling_width
      self.field_sampling_width = sampling_width
      self.amplitude_sampling_width = sampling_width
    self.target_acceptance = target_acceptance #TODO: hard code as fct of nuber of parameter space dims
    self.mean = np.array([]) #measuring that set of parameters that constitute the basis of parameter space in each case
    self.mean=self.construct_state(initial_amplitude, initial_field_coeffs)
    self.observables = self.construct_observables_state(initial_amplitude, initial_field_coeffs) 
    self.observabes_names = ["amplitude_squared"]
    self.params_names = ["abs_amplitude"]
    self.params_names.extend(["abs_c"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)]) 
                                                                #for output files etc, labelling coefficeints by their wavenumber index from -n to n, 
                                                                #even though they are stored in ordered list with implicit indices 0 to 2n+1

  def update_mean(self, state):
    assert (isinstance(state, np.ndarray))
    self.mean *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.mean += state / self.measure_step_counter


  def update_observables_mean(self, state_observables):
    self.observables *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.observables += state_observables / self.measure_step_counter


  def step_fieldcoeffs(self, field_coeffs, field_energy, amplitude, system,
                      amplitude_change=False):
    """
    """
    proposed_field_coeffs = self.draw_field_coeffs_from_proposal_distribution(amplitude, field_coeffs)
    #print("proposed field coeffs", proposed_field_coeffs, "old", field_coeffs)
    proposed_field_energy = system.calc_field_energy(field_coeffs=proposed_field_coeffs, amplitude=amplitude,
                                                     amplitude_change=False)
    accept= self.metropolis_decision(field_energy, proposed_field_energy)
    #if self.field_step_counter%60==0:
    #print("proposed field energy", proposed_field_energy, "old", field_energy)
    #print("amplitude", amplitude, "field coeffs", field_coeffs)
    if accept:
      #if self.field_step_counter%60==0:
      #print("accepted")
      field_energy = proposed_field_energy
      field_coeffs = proposed_field_coeffs
    self.update_field_sigma(accept)
    return field_coeffs, field_energy

  def step_amplitude(self, amplitude, field_coeffs, surface_energy, field_energy, system):
    """
    Stepping amplitude by metropolis algorithm.
    :param amplitude:
    :param field_coeffs:
    :param surface_energy:
    :param field_energy:
    :param system:
    :return:
    """
    proposed_amplitude = self.draw_amplitude_from_proposal_distriution(amplitude)
    #print(proposed_amplitude)
    if abs(proposed_amplitude) >= 1:
      # don't accept.
      # like an infinite energy barrier to self-intersection.
      # does not violate symmetric jump distribution, because this is like
      # an energy-landscape-based decision after generation
      self.update_amplitude_sigma(accept=False) #DO NOT have an effect on step length from this - leads to ever smaller steps that still get rejected half the time
      #print("early return")
      return amplitude, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    accept = self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy))
    #print("old energy ", field_energy, surface_energy)
    #print("new energys", new_field_energy, new_surface_energy)
    if accept:
      #print("accepted")
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
    self.update_amplitude_sigma(accept)
    #print("updated a sigma")
    return amplitude, surface_energy, field_energy

  def step_all(self, amplitude, field_coeffs, surface_energy, field_energy, system):
    """
    Step all parameters (amplitude, field coeffs) simultaneously.  True metropolis algorithm.
    Generic to any proposal distribution and adaptive algorithm
    :param amplitude:
    :param field_coeffs:
    :param surface_energy:
    :param field_energy:
    :param system: object which holds functions for calculating energy.  May store pre-calculated values and shortcuts.
    :return: new state and energy in the form tuple (ampltiude, field_coeffs (dict), surface_energy, field_energy).  Identical to input parameters if step was rejected.  Also modifies step counter and acceptance rate counter.
    """
    #print("field energy in", field_energy)
    proposed_amplitude, proposed_field_coeffs = self.draw_all_from_proposal_distribution(amplitude, field_coeffs)
    #print("propsed state", proposed_amplitude, proposed_field_coeffs)
    if abs(proposed_amplitude) >= 1:
      self.update_sigma(False)
      return amplitude, field_coeffs, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    accept = self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy))
    #print("step with", self.sampling_width, self.covariance_matrix[0,0])
    #print("surface energy", new_surface_energy, "old suface energy", surface_energy)
    #print("field_energy", new_field_energy, "old_field_energy", field_energy)
    #if self.step_counter%60==0:
      #print("proposed field energy", new_field_energy, "old", field_energy)
      #print("amplitude", amplitude)
    if accept:
      #if self.step_counter%60==0:
        #print("accepted")
      #print("accepted")
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      #self.accepted_counter += 1
    self.update_sigma(accept)
    # output system properties, energy
    #print("field_energy out ", field_energy)
    return amplitude, field_coeffs, surface_energy, field_energy


  def update_sigma(accept):
    pass
  def update_field_sigma(accept):
    pass
  def update_amplitude_sigma(accept):
    pass

  def measure(self,amplitude, field_coeffs):
    self.measure_step_counter +=1
    #basic: measure means of two goups of variables
    state = self.construct_state(amplitude, field_coeffs)
    self.update_mean(state)
    state_observables = self.construct_observables_state(amplitude, field_coeffs)
    self.update_observables_mean(state_observables)

  def draw_field_coeffs_from_proposal_distribution(self, amplitude, field_coeffs):
    """in basic abs(c_i) variable scheme - step ampltiude of each, vary phase by fixed amount"""
    state = self.construct_state(amplitude, field_coeffs)
    old_phases = map(cmath.phase,field_coeffs)
    proposed_addition = np.random.multivariate_normal(np.zeros(self.num_field_coeffs),
                                                      self.field_sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    proposed_state = state + proposed_addition
    proposed_field_coeffs = np.array([self.modify_phase(proposed_amplitude, old_phase) for (proposed_amplitude, old_phase) in  zip(proposed_state[1:], old_phases)])
    # as draw_all but discarding addition to amplitude, with field sampling width
    return proposed_field_coeffs

  def draw_field_coeff_from_proposal_distribution(self, field_coeff, index):
    old_phase = cmath.phase(field_coeff)
    amplitude = abs(field_coeff)
    index_in_cov_matrix = 1 + self.max_field_coeffs + index
    proposed_addition = random.gauss(0, self.field_sampling_width**2 * self.covariance_matrix[index_in_cov_matrix, index_in_cov_matrix])
    proposed_field_coeff = self.modify_phase(amplitude + proposed_addition, old_phase)
    return proposed_field_coeff

  def draw_amplitude_from_proposal_distriution(self, amplitude):
    proposed_amplitude = amplitude + random.gauss(0, self.amplitude_sampling_width**2 * self.covariance_matrix[0,0]) #technically multiply by + or - 1 depending on sign of amplitud, because parameter that covariance matrix applies to is abd(amplitude).  But not needed when not simultaneousy drawing in other parameters.
    return proposed_amplitude

  def draw_all_from_proposal_distribution(self, amplitude, field_coeffs):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    state = self.construct_state(amplitude, field_coeffs)
    old_phases = [cmath.phase(field_coeffs[key]) for key in field_coeffs]
    proposed_addition = np.random.multivariate_normal([0] * len(state),
                                                      self.sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    proposed_amplitude = amplitude + (-1 if amplitude < 0 else 1) * proposed_addition[0]
    proposed_field_coeff_amplitude = [original + addition for (original, addition) in
                                      zip(state[1:], proposed_addition[1:])]
    proposed_field_coeffs = dict(
      [(key, self.modify_phase(proposed_amplitude, old_phase)) for (key, (proposed_amplitude, old_phase)) in
       zip(self.keys_ordered, zip(proposed_field_coeff_amplitude, old_phases))])
    return proposed_amplitude, proposed_field_coeffs

  def metropolis_decision(self, old_energy, proposed_energy):
    """
    Considering energy difference and temperature, return decision to accept or reject step
    :param old_energy: current system total energy
    :param proposed_energy: system total energy after proposed change
    :return: bool True if step will be accepted; False to reject
    """
    diff = proposed_energy - old_energy
    #print("diff", diff, "temp", self.temp)
    assert (self.temp is not None)
    if diff <= 0:
      return True  # choice was made that 0 difference -> accept change
    elif diff > 0 and self.temp == 0:
      return False
    else:
      probability = math.exp(- 1 * diff / self.temp)
      #print("probability", probability)
      if random.uniform(0, 1) <= probability:
        return True
      else:
        return False

  def construct_state(self, amplitude, field_coeffs):
    """
    helper function to construct position in parameter space as np array 
    By default the parameters we track for correlation matrix, co-adjustemnt of step sizes are [abs(amplitude), {abs(field_coeff_i)}] (in this order)
    override in derived classes where different representation of the parameters is tracked.
    """
    state = [abs(amplitude)]  # TODO: abs?
    state.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    return np.array(state)


  def construct_observables_state(self, amplitude, field_coeffs):
    """
    helper function to construct current list of additional quantities whose mean is to be measured
    in base class: none
    """
    return np.array([amplitude**2])

  def modify_phase(self, amplitude, phase, phase_sigma=.1):
    """
    takes a random number and changes phase by gaussian proposal distribution
    :param phase_sigma: width of gaussian distribution to modify phase by.  Optional, default .5 (pi)
    """
    # TODO : pass option to set phase_sigma upwards
    new_phase = phase + random.gauss(0, phase_sigma)
    return cmath.rect(amplitude, new_phase)

  def random_complex(self, r):
    """
    a random complex number with a random (uniform between -pi and pi) phase and exactly the given ampltiude r
    :param r: magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation
    """
    phi = random.uniform(-math.pi, math.pi)
    return cmath.rect(r, phi)

  @staticmethod
  def gaussian_complex(sigma=.1): #by default v low chance of constructing random numbers magnitude>1  - because this leads to the v small field stepsize problem?
    """
    a random complex number with completely random phase and amplitude drawn from a gaussian distribution with given width sigma (default 1)
    :param sigma: width of gaussian distribution (centered around 0) of possible magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation 
    """
    amplitude = random.gauss(0, sigma)
    phase = random.uniform(0, 2 * math.pi)
    return cmath.rect(amplitude, phase)


  def set_temperature(self, new_temp):
    """
    change metropolis engine temperature
    :param new_temp: new temperature value, float >=0 .
    """

    assert (new_temp >= 0)
    self.temp = new_temp

class StaticCovarianceAdaptiveMetropolisEngine(MetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  here first without updating covariance matrix - assuming covariance is approximately the identity matrix (default) indicating no correlations, or a covariance matrix from known correlations of system parameters is given.
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=None, sampling_width=0.05, temp=0, covariance_matrix=None):
    super().__init__(initial_field_coeffs, initial_amplitude, sampling_width, temp, covariance_matrix)
    # alpha (a constant used later) := -phi^(-1)(target acceptance /2) 
    # ,where  phi is the cumulative density function of the standard normal distribution
    # norm.ppf ('percent point function') is the inverse of the cumulative density function of standard normal distribution
    self.alpha = -1 * scipy.stats.norm.ppf(self.target_acceptance / 2)
    self.m = self.param_space_dims
    self.steplength_c = None
    self.ratio = ( #  a constant - no need to recalculate 
        (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.alpha ** 2 / 2) / 2 * self.alpha + 1 / (
        self.m * self.target_acceptance * (1 - self.target_acceptance)))
    # inherit mean, construct mean from base class : abs amplitude, abs of field coeffs
    # inherit other observables to measure from base class: none

  def update_field_sigma(self, accept):
    m = self.param_space_dims//2 -1
    self.field_step_counter +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter / self.m, 200)) # because it shouldnt converge until cov is measured
    field_steplength_c = self.field_sampling_width * self.ratio
    if accept:
      #print("bigger field step")
      self.field_sampling_width += field_steplength_c * (1 - self.target_acceptance) / step_number_factor
      #print(self.field_sampling_width)
    else:
      self.field_sampling_width -= field_steplength_c * self.target_acceptance / step_number_factor
    assert (self.field_sampling_width) > 0

  def update_amplitude_sigma(self, accept):
    #print("in", accept)
    self.amplitude_step_counter +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter , 200)) #divided by m=1
    amplitude_steplength_c = self.amplitude_sampling_width * self.ratio
    if accept:
      self.amplitude_sampling_width += amplitude_steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.amplitude_sampling_width -= amplitude_steplength_c * self.target_acceptance / step_number_factor
      #print("decreasestep size")
    assert (self.amplitude_sampling_width) > 0

  def update_sigma(self, accept):
    self.step_counter +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter / self.m, 200))
    self.steplength_c = self.sampling_width * self.ratio
    if accept:
      self.sampling_width += self.steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.sampling_width -= self.steplength_c * self.target_acceptance / step_number_factor
    assert (self.sampling_width) > 0
  # TODO: test for convergence of sampling_width, c->0


class RobbinsMonroAdaptiveMetropolisEngine(StaticCovarianceAdaptiveMetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  without updating covariance matrix - assuming covariance is approximately identity matrix
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=0, sampling_width=0.05, temp=0, method="simultaneous",
               covariance_matrix=None):
    super().__init__(initial_field_coeffs=initial_field_coeffs, initial_amplitude=initial_amplitude,
                     sampling_width=sampling_width, temp=temp, covariance_matrix=covariance_matrix)
    # still working with the set of parameters : abs(amplitude), abs(field coeffs)
    # additional things to measure: none

  
  def measure(self, amplitude, field_coeffs):
    self.measure_step_counter +=1
    new_state = self.construct_state(amplitude, field_coeffs)
    old_mean = copy.copy(self.mean)
    self.update_mean(state=new_state)
    if self.measure_step_counter > 50: #down from recommendedation because when measurement is taken only every n steps
                                  # from more statistically independent data I expect to get a more accurate estimate of cov matrix sooner
      self.update_covariance_matrix(old_mean=old_mean, state=new_state)
    # other parameters that are not the basis for cov matrix
    state_observables = self.construct_observables_state(amplitude, field_coeffs)
    self.update_observables_mean(state_observables)

  def update_covariance_matrix(self, old_mean, state):
    i = self.measure_step_counter
    small_number = self.sampling_width ** 2 / self.measure_step_counter #stabilizes - coutnerintuitivelt causes nonzero estimated covariance for completely constant parameter at low stepcounts #self.step_counter instead of measurestpcounter for a smaller addition
    #print("old_mean", old_mean, "mean", self.mean, "state", state, "smallnumber", small_number)
    #print("added",  np.outer(old_mean,old_mean) - i/(i-1)*np.outer(self.mean,self.mean) + np.outer(state,state)/(i-1) + small_number*np.identity(self.param_space_dims))
    # print("multiplied", (i-2)/(i-1)*self.covariance_matrix)
    #print("result",  (i-2)/(i-1)*self.covariance_matrix +  np.outer(old_mean,old_mean)- i/(i-1)*np.outer(self.mean,self.mean) + np.outer(state,state)/(i-1) + small_number*np.identity(self.param_space_dims))
    self.covariance_matrix *= (i - 2) / (i - 1)
    self.covariance_matrix += np.outer(old_mean, old_mean) - i / (i - 1) * np.outer(self.mean, self.mean) + np.outer(
      state, state) / (i - 1) + small_number * np.identity(self.param_space_dims)
  # TODO: test for convergence of sampling_width, c->0


########## three schemes for more accurate covariance matrix ###########
class RealImgAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
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

class PhasesAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
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


class ComplexAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude=0, sampling_width=.05, covariance_matrix=None,
               temp=0):
    super().__init__(initial_field_coeffs=initial_field_coeffs, sampling_width=sampling_width, temp=temp,
                     covariance_matrix=covariance_matrix)
    #these text descriptions should match lists created in contruct_state, construct_observables_state functions
    self.param_space_dims = len(initial_field_coeffs) + 1
    if covariance_matrix is None:
      self.covariance_matrix = .2* (np.identity(self.param_space_dims , dtype = 'complex128'))
    self.m = self.param_space_dims
    
    #these text descriptions should match lists created in contruct_state, construct_observables_state functions
    self.params_names= ["amplitude"] # key to interpreting output: mean of parameters, covariance matrix
    self.params_names.extend(["c_"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)])
    self.observables_names = ["abs_amplitude"]
    self.observables_names.extend(["abs_c_"+str(i) for i in range(-self.max_field_coeffs, self.max_field_coeffs+1)])


  def construct_state(self, amplitude, field_coeffs):
    """
    should only be used once in init - we then pass the state in and out
    """
    state = [amplitude+0j] #amplitude as complex number, arbitrarily real
    state.extend([field_coeff for field_coeff in field_coeffs]) 
    return np.array(state)


  def construct_observables_state(self, amplitude=None, field_coeffs=None):
    """for arguments for compatibility of init with superclasses
    """
    observables_state = np.array(list(map(abs, self.mean))) # overly compliated way so that type is real
    return observables_state

  def construct_observables_state2(self, state):
    """ the one to use in rest of simulation
    """
    observables_state = abs(state) #should vectorized-apply if abs is a standard fct for np arrays
    return observables_state

  
  def measure(self, state):
    self.measure_step_counter +=1
    old_mean = copy.copy(self.mean)
    self.update_mean(state=state)
    if self.measure_step_counter > 50: #down from recommendedation because when measurement is taken only every n steps
                                  # from more statistically independent data I expect to get a more accurate estimate of cov matrix sooner
      self.update_covariance_matrix(old_mean=old_mean, state=state)
    # other parameters that are not the basis for cov matrix
    state_observables = self.construct_observables_state2(state)
    self.update_observables_mean(state_observables)

  def update_observables_mean(self, state_observables):
    self.observables *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.observables += state_observables / self.measure_step_counter
    #print("observables mean", type(self.observables[0]))

  def update_covariance_matrix(self, old_mean, state):
    i = self.measure_step_counter
    small_number = self.sampling_width ** 2 / i #stabilizes - coutnerintuitivelt causes nonzero estimated covariance for completely constant parameter at low stepcounts
    self.covariance_matrix *= (i - 2) / (i - 1)
    self.covariance_matrix += np.outer(old_mean, old_mean.conjugate()) - i / (i - 1) * np.outer(self.mean, self.mean.conjugate()) + np.outer(
        state, state.conjugate()) / (i - 1) + small_number  *np.identity(self.param_space_dims, dtype = 'complex128') #complex version TODO check


  def step_fieldcoeffs(self, state, field_energy, system,
                      amplitude_change=False):
    """
    """
    proposed_state = self.draw_field_coeffs_from_proposal_distribution(state)
    #print("proposed field coeffs", proposed_field_coeffs, "old", field_coeffs)
    proposed_field_energy = system.calc_field_energy(state= proposed_state,
                                                     amplitude_change=False)
    accept= self.metropolis_decision(field_energy, proposed_field_energy)
    #if self.field_step_counter%60==0:
    #print("proposed field energy", proposed_field_energy, "old", field_energy)
    #print("amplitude", amplitude, "field coeffs", field_coeffs)
    if accept:
      #if self.field_step_counter%60==0:
      #print("accepted")
      field_energy = proposed_field_energy
      state = proposed_state
    self.update_field_sigma(accept)
    return state, field_energy

  def multivariate_normal_complex(self, mean, covariance_matrix):
    #placeholder drawing seperately; ignores cross-correlations
    # add self.gaussian_complex(sigma) to each, with sigma from diagonal of cov matrix
    #diagonal of covariance matrix as 1d np array
    #print("in", mean, covariance_matrix)
    covariances = np.diagonal(covariance_matrix)
    #map self.gausian_complex over the list
    additions = np.array(list(map(self.gaussian_complex, covariances)))
    #add to mean
    return mean + additions


  def draw_field_coeffs_from_proposal_distribution(self, state):
    old_amplitude = state[0]
    proposed_state = self.multivariate_normal_complex(state,
                                                      self.field_sampling_width ** 2 * self.covariance_matrix)
    #return a full state amplitude + field coeffs, but with only field coeffs changed
    proposed_state[0] = old_amplitude
    return proposed_state



  def step_amplitude(self, state, field_energy, surface_energy, system):
    """
    Stepping amplitude by metropolis algorithm.
    :param amplitude:
    :param field_coeffs:
    :param surface_energy:
    :param field_energy:
    :param system:
    :return:
    """
    amplitude = state[0]
    proposed_amplitude = self.draw_amplitude_from_proposal_distriution(amplitude)
    if abs(proposed_amplitude) >= 1:
      # don't accept.
      # like an infinite energy barrier to self-intersection.
      # does not violate symmetric jump distribution, because this is like
      # an energy-landscape-based decision after generation
      self.update_amplitude_sigma(accept=False)
      return state, surface_energy, field_energy
    system.evaluate_integrals(proposed_amplitude)
    new_surface_energy = system.calc_surface_energy(amplitude = proposed_amplitude, amplitude_change=False)
    new_field_energy = system.calc_field_energy(state=state, amplitude_change=False) # will have the wrong amplitude at state[0] but that's ok if not used for change=False
    accept = self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy))
    #print("old energy ", field_energy, surface_energy)
    #print("proposed_ampltide", proposed_amplitude)
    #print("new energys", new_field_energy, new_surface_energy)
    if accept:
      #print("accepted")
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
    self.update_amplitude_sigma(accept)
    #print("updated a sigma")
    state[0]=amplitude
    #print(state[0])
    return state, surface_energy, field_energy
