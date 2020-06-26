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

  def __init__(self, energy_function, reject_condition = None, initial_real_params=None, initial_complex_params=None, sampling_width=0.05, covariance_matrix_real=None, covariance_matrix_complex=None, params_names=None, target_acceptance=.3, temp=0, energy_term_dependencies=None):
    """
    :param energy_function: function that calculates an energy value given parameter values : [real values] [complex values] -> float .  important, mandatory: metropolis engine is coupled to the desired physics problem by setting this function.
    :param initial_real_params: initial values for real parameters.  While the starting value is not so important, the length of this list is used to set up correct dimensions of paramter space.  Optional but either intial_real_params or initial_complex_params must be a list of length >=1.
    :param initial_complex_params: initial values for complex parameters.  While the starting value is not so important, the length of this list is used to set up correct dimensions of paramter space.  Optional but either intial_real_params or initial_complex_params must be a list of length >=1.
    :param params_names: list of strings [names of real parameters, names of complex parameters]. purely for user convenience - write them down to remember tham, use them to automatically label output data.  Optional, they will otherwise be set to "real_param_i", "complex_param_i" 
    :param sampling_width: initial width sigma of proposal distribution - recommended to input final value(s) from last simulation as an initial order of magnitude estimate.  single value or list of values for groups.  In non-adaptive base class, the initial value will be kept throughout the simulation.
    :param temp: system temperature in k_b T.  default 0.
    :param real_covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of real parameters.
    :param complex_covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of complex parameters. Must be complex-type np array. WARNING: at the moment only variances (the diagonal) is used, covariances between two different complex parameters are measured but using them to shape proposal distribution is not yet implemented.
    :param reject_condition: function list [real parameter values, complex parameter values] -> bool.  A system constraint: steps intot his region of parameter space are rejected.
    """
    
    #parameter space #
    if initial_real_params is None and initial_complex_params is None:
      print("must give list containing  at least one value for initial real or complex parameters")
      raise(ValueError)
    if initial_real_params is not None: 
      self.real_params = initial_real_params
      self.num_real_params = len(initial_real_params)
    else:
      self.real_params = []
      self.num_real_params = 0
      self.step_all = self.step_complex_group  # if the parameter space is all complex, go directly to step_complex_group when all
      self.measure = self.measure_complex_system
    if initial_complex_params is not None:
      self.complex_params = initial_complex_params
      self.num_complex_params = len(initial_complex_params)
    else:
      self.complex_params = []
      self.num_complex_params = 0
      self.step_all = self.step_real_group  # if the parameter space is all real, go directly t step_real_group  when stepping all
      self.measure = self.measure_real_system
    self.param_space_dims =  self.num_real_params  + self.num_complex_params
 
    # initializing quantities measured #
    if covariance_matrix_real is None and self.num_real_params >0:
      self.covariance_matrix_real = np.identity(self.num_real_params) 
    else:
      self.covariance_matrix_real = covariance_matrix_real
    if covariance_matrix_complex is None and self.num_complex_params>0 :
      self.covariance_matrix_complex = np.identity(self.num_complex_params, dtype = "complex128") 
    else:
      self.covariance_matrix_complex = covariance_matrix_complex
    #self.accepted_counter = 1
    self.step_counter = 1
    self.measure_step_counter = 1
    self.complex_group_step_counter =1
    self.real_group_step_counter =1
    self.group_step_counters=None
    self.real_mean=initial_real_params
    self.complex_mean = initial_complex_params
    #fill self.observables:
    self.construct_observables() # measured quatities derived from both real params and complex params are saved in one list, because they are all real datatype
    self.observables_mean = self.observables
    if params_names:
    	self.params_names = params_names
    else:
        self.params_names = ["param_"+str(i) for i in range(self.num_real_params)]
    self.observables_names =  ["abs_param_"+str(i) for i in range(self.num_real_params+self.num_complex_params)]
    self.observables_names.extend(["param_"+str(i)+"_squared" for i in range(self.num_real_params+self.num_complex_params)])

    # temporary variables of loop
    self.energy = None

    # metropolis step, proposal distribution, and adaptiveness
    self.temp = temp
    assert (self.temp is not None and self.temp >= 0)
    if isinstance(sampling_width, list): #TODO : conform this to how two groups are handles
      self.real_group_sampling_width = sampling_width[0]
      self.complex_group_sampling_width = sampling_width[1] #arbitrarily use this one when generating a small number to stabilize cov matrix
    else:
      self.sampling_width = sampling_width
      self.real_group_sampling_width = sampling_width
      self.complex_group_sampling_width = sampling_width
    self.target_acceptance = target_acceptance #TODO: hard code as fct of nuber of parameter space dims

    #functions#
    if isinstance(energy_function, dict):
      self.calc_energy = energy_function
      if energy_term_dependencies:
      	self.energy_term_dependencies = energy_term_dependencies 
      else: 
        self.energy_term_dependencies = dict([(name,["complex", "real"]) for name in energy_function])
      self.complex_group_energy_terms = [name for name in energy_term_dependencies if "complex" in energy_term_dependencies[name]]#which energy terms change when complex params change
      self.real_group_energy_terms = [name for name in energy_term_dependencies if "real" in energy_term_dependencies[name]]
      self.calc_energy_total = 0#difficulat function construction
    else:
      self.calc_energy ={"total": energy_function}
      self.calc_energy_total = energy_function #use in step_all
      if energy_term_dependencies:
        print("dependecies of term-wise energy functions given, but energy fcuntion was not given term-wise")
      #used in case energy is a single term and parameter space is only real or only complex
      self.real_group_energy_terms = ["total"]
      self.complex_group_energy_terms = ["total"]
    if reject_condition is None:
      self.reject_condition = lambda real_params, complex_params: False  # by default no constraints



  def set_energy_function(self, energy_function):
    #by default this is also called for system energy on change of real group only or complex group only
    self.calc_energy = energy_function

  
  
  """ do groups later
  def set_groups(self,group_members, group_energy_terms):
    print("set self.group_members")
    self.group_members = group_members # dict (group id : [positions in state])
    self.group_energy_terms = group_energy_terms #dict (group id: [energy term ids]) 
    self.group_sampling_width = dict([(group_id, self.sampling_width) for group_id in group_members])
    print("set goup sampling widths", self.group_sampling_width)
    self.group_step_counter = dict([(group_id, 0) for group_id in group_members])
  
  def set_energy_terms(self,energy_terms, dependencies):
    self.calc_energy_terms = energy_terms  #term id : function
    self.dependencies = dependencies # term id: [groups ids ]
  """

  def set_reject_condition(self, reject_fct):
    """
    set a function that takes np array state and returns a bool as system constraint / like infintie energy barrier
    """
    self.reject_condition = reject_fct

  def set_initial_sampling_width(self, sampling_width):
    self.group_sampling_width = sampling_width


  
  ######################## metropolis step ####################################

  """
  def step_group(self, group_id, state, energy_list):
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
  """

  def step_complex_group(self):
    proposed_complex_params = self.draw_complex_group()
    if self.reject_condition(self.real_params, proposed_complex_params):
      self.update_complex_group_sigma(accept=False)
      #do nothing towards updating real_params, complex_params, energy_dict
      return False
    #self.complex_group_energy_terms : keys to energy terms which change when complex group params change
    energy_partial = sum([self.energy[term_id] for term_id in self.complex_group_energy_terms]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy[term_id](self.real_params, proposed_complex_params)) for term_id in self.complex_group_energy_terms])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    #print("state", state,  "proposed state", proposed_state, "changed energy terms", proposed_energy_terms, "former value", energy_partial)
    if accept:
      #print("accepted")
      for term_id in self.complex_group_energy_terms:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      #print("new energy", energy_list)
      self.complex_params = proposed_complex_params
    self.update_complex_group_sigma(accept)
    return accept


  def step_real_group(self):
    proposed_real_params = self.draw_real_group()
    if self.reject_condition(proposed_real_params, self.complex_params):
      self.update_real_group_sigma(accept=False)
      return False
    #self.real_group_energy_terms : keys to energy terms which change when real-group params change
    energy_partial = sum([self.energy[term_id] for term_id in self.real_group_energy_terms]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy[term_id](proposed_real_params, self.complex_params)) for term_id in self.real_group_energy_terms])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    #print("state", state,  "proposed state", proposed_state, "changed energy terms", proposed_energy_terms, "former value", energy_partial)
    if accept:
      #print("accepted")
      for term_id in self.real_group_energy_terms:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      self.real_params = proposed_real_params
    self.update_real_group_sigma(accept)
    return accept


  def step_all(self):
    """
    Step all parameters (amplitude, field coeffs) simultaneously. 
    Generic to any proposal distribution (draw function) and adaptive algorithm
    """
    proposed_real_params, proposed_complex_params = self.draw_all()
    if self.reject_condition(proposed_real_params, proposed_complex_params):
      self.update_sigma(accept=False) 
      return False
    proposed_energy = self.calc_energy_total(proposed_real_params, proposed_complex_params)
    #print("energy", energy, "proposed_energy", proposed_energy)
    accept = self.metropolis_decision(self.energy, proposed_energy)
    if accept:
      #print("accepted", proposed_state, proposed_energy)
      self.energy = proposed_energy # TODO : saved at energy_dict because I dont expect a switch of method within a simulation
      state = proposed_state
    self.update_sigma(accept)
    return accept

  def draw_real_group(self):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    """
    # no arguments - look at self.real_params
    proposed_state = np.random.multivariate_normal(self.real_params,
                                                      self.real_group_sampling_width ** 2 * self.covariance_matrix_real,
                                                      check_valid='raise')
    return proposed_state

  def draw_complex_group(self):
    covariances = np.diagonal(self.covariance_matrix_complex)
    #map self.gausian_complex over the list
    addition_complex = np.array(list(map(lambda sc : self.gaussian_complex(sc[0]**2*sc[1]), (self.complex_group_sampling_width, covariances) ))) #addition has compeletely random phase
    return np.array([np.random.normal(m, self.sampling_width[index] * covariances[index]) if index < self.num_real_params else m + addition_complex[index] for index,m in enumerate(state) ])
  
  def draw_all(self):
    return draw_real_group(), draw_complex_group()
  """
  def draw_group(self, group_id, state):
    # TODO  : rethinnk this
    # draw real andor complex, then combine updated values in desired positions with state in
    proposed_state = np.random.multivariate_normal(state, self.group_sampling_width[group_id] ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    #return re-drwn values for members of group, original values for others
    for index in self.group_members[group_id]:
      new_state = copy.copy(state)
      new_state[index] = proposed_state[index]
    return new_state
  """
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
 
  def measure(self):
    self.measure_step_counter +=1
    self.measure_real()
    self.measure_complex()
    # other parameters that are not the basis for cov matrices:
    self.construct_observables()
    self.update_observables_mean()

  #only used when parameter space is pure real or pure complex.   
  def measure_real_system(self):
    self.measure_step_counter +=1
    self.measure_real()
    # other parameters that are not the basis for cov matrices:
    self.construct_observables()
    self.update_observables_mean()

  def measure_complex_system(self):
    self.measure_step_counter +=1
    self.measure_complex()
    # other parameters that are not the basis for cov matrices:
    self.construct_observables()
    self.update_observables_mean()

  def measure_real(self):
    #new_state = self.construct_state(amplitude, field_coeffs)
    old_mean_real = copy.copy(self.real_mean)
    self.update_real_mean()
    if self.measure_step_counter > 50: #down from recommendedation because when measurement is taken only every n steps
                                  # from more statistically independent data I expect to get a more accurate estimate of cov matrix sooner
      self.update_covariance_matrix_real(old_mean_real)
 
  def measure_complex(self):
    old_mean_complex = copy.copy(self.complex_mean)
    self.update_complex_mean()
    if self.measure_step_counter > 50: #down from recommendedation because when measurement is taken only every n steps
                                  # from more statistically independent data I expect to get a more accurate estimate of cov matrix sooner
      self.update_covariance_matrix_complex(old_mean_complex)
 
  def update_mean(self):
    self.update_real_mean()
    self.update_complex_mean()

  def update_real_mean(self):
    assert (isinstance(self.real_params, np.ndarray))
    self.real_mean *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.real_mean += self.real_params / self.measure_step_counter

  def update_complex_mean(self):
    self.complex_mean *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.complex_mean += self.complex_params / self.measure_step_counter

  def update_observables_mean(self):
    self.observables_mean *= (self.measure_step_counter - 1) / self.measure_step_counter
    self.observables_mean += self.observables / self.measure_step_counter

  def update_covariance_matrix_real(self, old_mean_real):
    i = self.measure_step_counter
    small_number = self.real_group_sampling_width ** 2 / self.measure_step_counter #stabilizes - coutnerintuitivelt causes nonzero estimated covariance for completely constant parameter at low stepcounts #self.step_counter instead of measurestpcounter for a smaller addition
    self.covariance_matrix_real *= (i - 2) / (i - 1)
    self.covariance_matrix_real += np.outer(old_mean_real, old_mean_real) - i / (i - 1) * np.outer(self.real_mean, self.real_mean) + np.outer(
      self.real_params, self.real_params) / (i - 1) + small_number * np.identity(self.num_real_params)
  def update_covariance_matrix_complex(self, old_mean_complex):
    i = self.measure_step_counter
    small_number = self.complex_group_sampling_width ** 2 / self.measure_step_counter
    self.covariance_matrix_real *= (i - 2) / (i - 1)
    self.covariance_matrix_real += np.outer(old_mean_complex, old_mean_complex.conjugate()) - i / (i - 1) * np.outer(self.complex_mean, self.complex_mean.conjugate()) + np.outer(
      self.complex_params, self.complex_params.conjugate()) / (i - 1) + small_number*np.identity(self.num_complex_params, dtype = 'complex128')


  def update_sigma(self, accept):
    """
    Does nothing in non-adaptive base class,
    override in other classes
    """
    pass
  def update_real_group_sigma(self,accept):
    """
    Does nothing in non-adaptive base class,
    override in other classes
    """
    pass
  def update_complex_group_sigma(self,accept):
    pass

 
  def construct_observables(self):
    # TODO : faster method?
    observables = [abs(x) for x in self.real_params]
    observables.extend([abs(x) for x in self.complex_params])
    observables.extend([x**2 for x in self.real_params])
    observables.extend([x*x.conjugate() for x in self.complex_params])
    self.observables = np.array(observables)


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
    if params_names:
    	self.params_names = params_names
    else:
        self.params_names = ["param_"+str(i) for i in range(self.num_real_params+self.num_complex_params)]
    self.observables_names =  ["abs_param_"+str(i) for i in range(self.num_real_params+self.num_complex_params)]
    self.observables_names =  ["abs_param_"+str(i) for i in range(self.num_real_params+self.num_complex_params)]
    

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
    small_number = self.sampling_width[0] ** 2 / i #stabilizes - coutnerintuitivelt causes nonzero estimated covariance for completely constant parameter at low stepcounts
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
    addition_complex = np.array(list(map(lambda sc : self.gaussian_complex(sc[0]**2*sc[1]), (self.sampling_width, covariances) ))) #addition has compeletely random phase
    #add to mean

    return np.array([np.random.normal(m, self.sampling_width[index] * covariances[index]) if index < self.num_real_params else m + addition_complex[index] for index,m in enumerate(state) ])

  def draw_group(self, group_id, state):
    # TODO  : rethinnk this
    new_values = self.draw_all(state)
    return np.array([new_value if index in self.group_members[group_id] else state[index] for index, new_value in enumerate(new_values)])


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
import cmath
