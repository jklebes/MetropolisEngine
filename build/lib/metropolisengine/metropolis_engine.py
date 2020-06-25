import cmath
import copy
import math
import numpy as np
import random
import scipy.stats


class MetropolisEngine():
  """
  base class: non-adaptive Metropolis algorithm
  proposal distribution is fixed in width (sigma) and covariance.
  By default the covariance matrix of the multivariate proposal distribution is the identity matrix, equivalent to drawing from n independent gaussian distribution.
  A different fixed covariance matrix can alo be chosen.
  No adaptiveness guarantees ergodic sampling.
  Subclass StaticCovarianceAdaptiveMetropolisEngine is recommended minimum level of adaptiveness.
  """

  def __init__(self, initial_real_params, sampling_width=0.05, covariance_matrix=None, params_names=None,
               target_acceptance=.3, temp=0):
    """
    :param initial_field_coeffs: initial values for complex parameters
    :param sampling_width: float, width of proposal distribution.  A factor scaling covariance matrix in multivariate gaussian proposal distribution.
    :param temp: system temperature.  Optional, default 0.
    :param covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of free parameters: in general number of field coefficients + 1.  2*number of field coefficients+1 Where phase/ampltitude or real/imag parts of field coefficients are treated as separate degrees of freedom.  complex entries in subclass ComplexAdaptiveMetropolisEngine.
    """
    self.temp = temp
    assert (self.temp is not None and self.temp >= 0)
    self.num_real_params = len(initial_real_params)
    if not self.num_real_params:
      print("must give list containing  at least one value for initial real or complex parameters")
      raise(ValueError)
    self.param_space_dims =  self.num_real_params  
    if covariance_matrix is None:
      self.covariance_matrix = np.identity(self.param_space_dims) 
    else:
      self.covariance_matrix = covariance_matrix
    #self.accepted_counter = 1
    self.step_counter = 1
    self.measure_step_counter = 1
    self.group_step_counters=None
    if isinstance(sampling_width,dict): #TODO : conform this to how two groups are handles
      self.group_sampling_width = sampling_width
      self.sampling_width = self.field_sampling_width #arbitrarily use this one when generating a small number to stabilize cov matrix
    else:
      self.sampling_width = sampling_width
    self.target_acceptance = target_acceptance #TODO: hard code as fct of nuber of parameter space dims
    self.mean=self.construct_state(initial_real_params, []) # TODO - fix this hack by taking initial params out of init
    self.observables = self.construct_observables_state(initial_real_params, [])
    # TODO: should be input 
    if params_names:
    	self.params_names = params_names
    else:
        self.params_names = ["real_param_"+str(i) for i in range(self.num_real_params)]
    self.observables_names =  ["abs_param_"+str(i) for i in range(self.num_real_params)]
    self.observables_names.extend(["param_"+str(i)+"_squared" for i in range(self.num_real_params)])
    self.reject_condition = lambda state: False

  def set_energy_function(self, energy_function):
    #by default this is also called for system energy on change of real group only or complex group only
    self.calc_energy = energy_function

  def set_groups(self,group_members, group_energy_terms):
    """in : dict (group name: positions in state)"""
    print("set self.group_members")
    self.group_members = group_members # dict (group id : [positions in state])
    self.group_energy_terms = group_energy_terms #dict (group id: [energy term ids]) 
    self.group_sampling_width = dict([(group_id, self.sampling_width) for group_id in group_members])
    print("set goup sampling widths", self.group_sampling_width)
    self.group_step_counter = dict([(group_id, 0) for group_id in group_members])

  def set_energy_terms(self,energy_terms, dependencies):
    """energy-terms: dict (functions : dependencies on groups)"""
    self.calc_energy_terms = energy_terms  #term id : function
    self.dependencies = dependencies # term id: [groups ids ]
  
  def set_reject_condition(self, reject_fct):
    """
    set a function that takes np array state and returns a bool as system constraint / like infintie energy barrier
    """
    self.reject_condition = reject_fct
  
  ######################## metropolis step ####################################


  def step_group(self, group_id, state, energy_list):
    """
    Stepping amplitude by metropolis algorithm.
    """
    proposed_state = self.draw_group(group_id, state)
    if self.reject_condition(state):
      self.update_amplitude_sigma(accept=False)
      return state, energy_list
    energy_term_ids = self.group_energy_terms[group_id]
    energy_partial = sum([energy_list[term_id] for term_id in energy_term_ids]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id,self.calc_energy_terms[term_id](proposed_state)) for term_id in energy_term_ids] )
    proposed_energy_partial = sum(proposed_energy_terms.values()) #lookup of function for the partial energy - only all terms involving
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    #print("state", state,  "proposed state", proposed_state, "changed energy terms", proposed_energy_terms, "former value", energy_partial)
    if accept:
      #print("accepted")
      for term_id in energy_term_ids:
      	energy_list[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      #print("new energy", energy_list)
      state = proposed_state
    self.update_group_sigma(group_id,accept)
    return state, energy_list

  def step_all(self, state, energy):
    """
    Step all parameters (amplitude, field coeffs) simultaneously. 
    Generic to any proposal distribution (draw function) and adaptive algorithm
    """
    proposed_state = self.draw_all(state)
    #print("state", state, "proposed state", proposed_state)
    if self.reject_condition(proposed_state):
      self.update_sigma(accept=False) 
      return real_params, energy
    proposed_energy = self.calc_energy(proposed_state)
    #print("energy", energy, "proposed_energy", proposed_energy)
    accept = self.metropolis_decision(energy, proposed_energy)
    if accept:
      #print("accepted", proposed_state, proposed_energy)
      energy = proposed_energy
      state = proposed_state
    self.update_sigma(accept)
    return state, energy

  def draw_all(self, state):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    proposed_state = np.random.multivariate_normal(state,
                                                      self.sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    return proposed_state

  def draw_group(self, group_id, state):
    # TODO  : rethinnk this
    proposed_state = np.random.multivariate_normal(state, self.group_sampling_width[group_id] ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    #return re-drwn values for members of group, original values for others
    for index in self.group_members[group_id]:
      new_state = copy.copy(state)
      new_state[index] = proposed_state[index]
    return new_state

  def metropolis_decision(self, old_energy, proposed_energy):
    """
    Considering energy difference and temperature, return decision to accept or reject step
    :param old_energy: current system total energy
    :param proposed_energy: system total energy after proposed change
    :return: bool True if step will be accepted; False to reject
    """
    diff = proposed_energy - old_energy
    assert (self.temp is not None)
    if diff <= 0:
      return True  # choice was made that 0 difference -> accept change
    elif diff > 0 and self.temp == 0:
      return False
    else:
      probability = math.exp(- 1 * diff / self.temp)
      if random.uniform(0, 1) <= probability:
        return True
      else:
        return False

  ###################### measuring mean and covariance estimate #######################
 
  def measure(self, new_state):
    self.measure_step_counter +=1
    #new_state = self.construct_state(amplitude, field_coeffs)
    old_mean = copy.copy(self.mean)
    self.update_mean(state=new_state)
    if self.measure_step_counter > 50: #down from recommendedation because when measurement is taken only every n steps
                                  # from more statistically independent data I expect to get a more accurate estimate of cov matrix sooner
      self.update_covariance_matrix(old_mean=old_mean, state=new_state)
    # other parameters that are not the basis for cov matrix
    state_observables = self.construct_observables_state(new_state)
    self.update_observables_mean(state_observables)
  
  def update_mean(self, state):
    assert (isinstance(state, np.ndarray))
    #print("self.mean", self.mean, "state", state)
    self.mean *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.mean += state / self.measure_step_counter

  def update_observables_mean(self, state_observables):
    self.observables *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.observables += state_observables / self.measure_step_counter

  def update_covariance_matrix(self, old_mean, state):
    i = self.measure_step_counter
    small_number = self.sampling_width ** 2 / self.measure_step_counter #stabilizes - coutnerintuitivelt causes nonzero estimated covariance for completely constant parameter at low stepcounts #self.step_counter instead of measurestpcounter for a smaller addition
    self.covariance_matrix *= (i - 2) / (i - 1)
    self.covariance_matrix += np.outer(old_mean, old_mean) - i / (i - 1) * np.outer(self.mean, self.mean) + np.outer(
      state, state) / (i - 1) + small_number * np.identity(self.param_space_dims)
    #print("updated convariance", self.covariance_matrix)


  def construct_state(self, real_params, complex_params=None):
    """
    helper function to construct position in parameter space as np array 
    By default the parameters we track for correlation matrix, co-adjustemnt of step sizes are the raw values of real and complex params
    override in derived classes where different representation of the parameters is tracked.
    """
    return np.array(real_params)


  def construct_observables_state(self, real_params, complex_params=None):
    """
    helper function to construct current list of additional quantities whose mean is to be measured
    absolute values and squares of parameters (real and complex)
    If you want different values measured, make a metropolis engine subclass or otherwise override this
    """
    observables = [abs(param) for param in real_params]
    observables.extend([param**2 for param in real_params])
    return np.array(observables)

  def update_sigma(self, accept):
    """
    Does nothing in non-adaptive base class,
    override in other classes
    """
    pass
  def update_group_sigma(self, group_id, accept):
    """
    Does nothing in non-adaptive base class,
    override in other classes
    """
    pass
 
  ##############functions for complex number handling #########################

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
    a random complex number with a random (uniform between -pi and pi) phase and exactly the given magnitude r
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


class AdaptiveMetropolisEngine(MetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  """
  def __init__(self, initial_real_params, sampling_width=0.05, covariance_matrix=None, params_names=None, target_acceptance=.3, temp=0):
    super().__init__(initial_real_params, sampling_width=sampling_width,covariance_matrix=covariance_matrix, params_names=params_names, 
target_acceptance=target_acceptance, temp=temp)
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

  def update_group_sigma(self, group_id, accept):
    m = self.param_space_dims//2 -1 # TODO move to init?
    self.group_step_counter[group_id] +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter / self.m, 200)) # because it shouldnt converge until cov is measured
    field_steplength_c = self.group_sampling_width[group_id] * self.ratio
    if accept:
      #print("bigger field step")
      self.group_sampling_width[group_id] += field_steplength_c * (1 - self.target_acceptance) / step_number_factor
      #print(self.field_sampling_width)
    else:
      self.group_sampling_width[group_id] -= field_steplength_c * self.target_acceptance / step_number_factor
    assert (self.group_sampling_width[group_id]) > 0

  def update_sigma(self, accept):
    self.step_counter +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter / self.m, 200))
    self.steplength_c = self.sampling_width * self.ratio
    if accept:
      self.sampling_width += self.steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.sampling_width -= self.steplength_c * self.target_acceptance / step_number_factor
    assert (self.sampling_width) > 0
    #print("sampling width update to", self.sampling_width)



class ComplexAdaptiveMetropolisEngine(AdaptiveMetropolisEngine):
  # same as adaptive metropolis engine, but covariance matrix, state must be complex type
                                             # inflexible type because numpy arrays
  # and number of degrees of freedom for m, target acceptance ratio is possibly bigger
  def __init__(self, initial_real_params, initial_complex_params, sampling_width=.05, covariance_matrix=None, params_names = None, target_acceptance=.3,
               temp=0):
    super().__init__(initial_real_params=initial_real_params, sampling_width=sampling_width, covariance_matrix = None, params_names = params_names, target_acceptance = target_acceptance, temp=temp)
    #now add / redo anything having to do with complex type
    self.num_complex_params = len(initial_complex_params)
    self.param_space_dims =  self.num_real_params  + self.num_complex_params
    if covariance_matrix is None:
      self.covariance_matrix = (np.identity(self.param_space_dims , dtype = 'complex128'))
    else:
      # TODO : check given covariance matrix is complex or make complex
      self.covariance_matrix = covariance_matrix
    self.m = self.param_space_dims
    self.mean=self.construct_state(initial_real_params, initial_complex_params)
    self.observables = self.construct_observables_state(initial_real_params, initial_complex_params)
    

  def construct_state(self, real_params, complex_params):
    """
    should only be used once in init - we then pass the state in and out
    """
    state = [param+0j for param in real_params] #real params as complex number, phase means nothing
    state.extend([param for param in complex_params]) 
    return np.array(state)


  def construct_observables_state(self, amplitude=None, field_coeffs=None):
    """for arguments for compatibility of init with superclasses
    """
    observables_state = np.array(list(map(abs, self.mean))) # this chaining ensures that type is real
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


  def draw_all(self, state):
    # because there is no complex draw from multivariate gaussian distribution
    #placeholder drawing seperately; ignores cross-correlations
    # add self.gaussian_complex(sigma) to each, with sigma from diagonal of cov matrix
    #diagonal of covariance matrix as 1d np array
    #print("in", mean, covariance_matrix)
    covariances = np.diagonal(self.covariance_matrix)
    #map self.gausian_complex over the list
    addition_complex = np.array(list(map(lambda x : self.gaussian_complex(self.sampling_width**2*x), covariances))) #addition has compeletely random phase
    #add to mean

    return np.array([np.random.normal(m, self.sampling_width * covariances[index]) if index < self.num_real_params else m + addition_complex[index] for index,m in enumerate(state) ])

  def draw_group(self, group_id, state):
    # TODO  : rethinnk this
    proposed_state = np.random.multivariate_normal(state, self.group_sampling_width[group_id] ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    #return re-drwn values for members of group, original values for others
    for index in self.group_members[group_id]:
      new_state = copy.copy(state)
      new_state[index] = proposed_state[index]
    return new_state


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
 
  def step_amplitude_no_field(self, amplitude, surface_energy, system):
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
    if abs(proposed_amplitude) >= 1:
      # don't accept.
      # like an infinite energy barrier to self-intersection.
      # does not violate symmetric jump distribution, because this is like
      # an energy-landscape-based decision after generation
      self.update_amplitude_sigma(accept=False)
      return state, surface_energy, field_energy
    system.evaluate_integrals(proposed_amplitude)
    new_surface_energy = system.calc_surface_energy(amplitude = proposed_amplitude, amplitude_change=False)
    #new_field_energy = system.calc_field_energy(state=state, amplitude_change=False) # will have the wrong amplitude at state[0] but that's ok if not used for change=False
    accept = self.metropolis_decision(surface_energy,new_surface_energy)
    #print("old energy ", field_energy, surface_energy)
    #print("proposed_ampltide", proposed_amplitude)
