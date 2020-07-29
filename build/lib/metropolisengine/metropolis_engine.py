import cmath
import copy
import math
import numpy as np
import random
import scipy.stats
import pandas
import metropolisengine.statistics

class MetropolisEngine():
  """
  Adaptive metropolis algorithm over a parameter space real and complex paramters

  adaptiveness ("update_sigma" functions) adapted from from Garthwaite, Fan & Sisson https://arxiv.org/abs/2006.12668
  """

  def __init__(self, energy_functions, reject_condition = None, initial_real_params=None, initial_complex_params=None, sampling_width=0.05, covariance_matrix_real=None, covariance_matrix_complex=None, params_names=None, target_acceptance=.3, temp=0, complex_sample_method = "multivariate-gaussian"):
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
     #initialize data collection
    self.real_params_time_series = []
    self.complex_params_time_series = []
    self.observables_time_series = []
    self.real_group_sampling_width_time_series =[]
    self.complex_group_sampling_width_time_series = []
    #parameter space #
    if initial_real_params is None and initial_complex_params is None:
      print("must give list containing  at least one value for initial real or complex parameters")
      raise(ValueError)
    if initial_real_params is not None: 
      self.real_params = np.array(initial_real_params)
      self.num_real_params = len(initial_real_params)
    else:
      self.real_params = np.array([])
      self.num_real_params = 0
      self.step_all = self.step_complex_group  # if the parameter space is all complex, go directly to step_complex_group when all
      self.measure = self.measure_complex_system
      self.real_params_time_series = None
      self.real_group_sampling_width_time_series = None
    if initial_complex_params is not None:
      self.complex_params = np.array(initial_complex_params)
      self.num_complex_params = len(initial_complex_params)
    else:
      self.complex_params =np.array([])
      self.num_complex_params = 0
      self.step_all = self.step_real_group  # if the parameter space is all real, go directly t step_real_group  when stepping all
      self.measure = self.measure_real_system
      self.complex_params_time_series = None
      self.complex_group_sampling_width_time_series = None
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
    self.real_mean=self.real_params
    self.complex_mean = self.complex_params
    #fill self.observables:
    self.construct_observables() # measured quatities derived from both real params and complex params are saved in one list, because they are all real datatype
    self.observables_mean = self.observables
    if params_names:
    	self.params_names = params_names
    else:
        self.params_names = ["param_"+str(i) for i in range(self.num_real_params+ self.num_complex_params)]
    self.observables_names =  ["abs_param_"+str(i) for i in range(self.num_real_params+self.num_complex_params)]
    self.observables_names.extend(["param_"+str(i)+"_squared" for i in range(self.num_real_params)])


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
    #adativeness
    self.target_acceptance = target_acceptance #TODO: hard code as fct of nuber of parameter space dims
    self.alpha = -1 * scipy.stats.norm.ppf(self.target_acceptance / 2)
    self.m = self.param_space_dims
    self.steplength_c = None
    self.ratio = ( #  a constant - no need to recalculate 
        (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.alpha ** 2 / 2) / 2 * self.alpha + 1 / (
        self.m * self.target_acceptance * (1 - self.target_acceptance)))
    self.df = None #essentially a flag that final stats have not been gathered

    #functions#
    if isinstance(energy_functions, dict):
      self.calc_energy = energy_functions
      self.energy_term_names = set([]) #collect all energy term names 
      for keys in energy_functions.values():
        self.energy_term_names =  self.energy_term_names.union(keys)
    else:
      #if only a single function for system total energy was given
      self.energy_term_names = ["total"]
      self.calc_energy ={"complex":{"total": energy_functions} , "real":{"total": energy_functions}, "all":{"total": energy_functions}}
      self.calc_energy_total = energy_functions #use in step_all
    #initialize energy
    #self.energy:
    self.initialize_energy_dict() 
    self.energy_time_series = dict([(key,[]) for key in self.energy_term_names])    
    self.energy_total= self.calc_energy_total(initial_real_params, initial_complex_params)
    if reject_condition is None:
      self.reject_condition = lambda real_params, complex_params: False  # by default no constraints
    #switch out complex sample method if two-step, random phase sampling is requested
    if complex_sample_method == "magnitude-phase":
      self.step_complex_group = self.step_complex_group_magnitude_phase
    elif complex_sample_method != "multivariate-gaussian":
      print("complex_sample_method", complex_sample_method, "not recognized")
      print("defaulting to multivariate-gaussian")


  def set_energy_function(self, energy_function):
    self.calc_energy = energy_function
    self.energy_term_names = set([]) #collect all energy term names 
    for keys in energy_function.values():
      self.energy_term_names =  self.energy_term_names.union(keys)

  def set_reject_condition(self, reject_fct):
    """
    set a function that takes np array state and returns a bool as system constraint / like infintie energy barrier
    """
    self.reject_condition = reject_fct

  def set_initial_sampling_width(self, sampling_width):
    self.group_sampling_width = sampling_width


  def initialize_energy_dict(self):
    self.energy={}
    for key in self.energy_term_names:
      self.energy[key] = self.calc_energy["all"][key](self.real_params,self.complex_params)
    print("initialized energy dict, self.energy = ", self.energy)

  def calc_energy_total(self, proposed_real_params, proposed_complex_params):
    total = 0
    for key in self.energy_term_names:
      total += self.calc_energy["all"][key](proposed_real_params, proposed_complex_params)
    return total


  
  ######################## metropolis step ####################################

  def step_complex_group_magnitude_phase(self):
    """
    alternative scheme to stepping complex variables with complex multivariate gaussian proposal distribution
    stepping magnitude (from gaussian dist), phases (from uniform dist) in two steps
    - faster to equilibrium and wider sampling, generally worth it for complax params with weak phase correlation
    set metropolis enigne to use this one via "complex_sample_method" variable
    """
    self.step_complex_group_magnitude()
    self.step_complex_group_phase()

  def step_complex_group_magnitude(self):
    proposed_complex_params = self.draw_complex_magnitudes()
    if self.reject_condition(self.real_params, proposed_complex_params):
      self.update_complex_group_sigma(accept=False)
      return False
    energy_partial = sum([self.energy[term_id] for term_id in self.calc_energy["complex"]]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy["complex"][term_id](self.real_params, proposed_complex_params)) for term_id in self.calc_energy["complex"]])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    if accept:
      for term_id in self.calc_energy["complex"]:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      self.complex_params = proposed_complex_params
    self.update_complex_group_sigma(accept)
    return accept

  def step_complex_group_phase(self):
    proposed_complex_params = self.draw_complex_phases()
    if self.reject_condition(self.real_params, proposed_complex_params):
      self.update_complex_group_sigma(accept=False)
      return False
    #self.complex_group_energy_terms : keys to energy terms which change when complex group params change
    energy_partial = sum([self.energy[term_id] for term_id in self.calc_energy["complex"]]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy["complex"][term_id](self.real_params, proposed_complex_params)) for term_id in self.calc_energy["complex"]])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    if accept:
      for term_id in self.calc_energy["complex"]:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      self.complex_params = proposed_complex_params
    self.update_complex_group_sigma(accept)
    return accept

  def step_complex_group(self):
    proposed_complex_params = self.draw_complex_group()
    if self.reject_condition(self.real_params, proposed_complex_params):
      self.update_complex_group_sigma(accept=False)
      return False
    energy_partial = sum([self.energy[term_id] for term_id in self.calc_energy["complex"]]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy["complex"][term_id](self.real_params, proposed_complex_params)) for term_id in self.calc_energy["complex"]])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    if accept:
      for term_id in self.calc_energy["complex"]:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      self.complex_params = proposed_complex_params
    self.update_complex_group_sigma(accept)
    return accept

  def step_real_group(self):
    proposed_real_params = self.draw_real_group()
    if self.reject_condition(proposed_real_params, self.complex_params):
      self.update_real_group_sigma(accept=False)
      return False
    energy_partial = sum([self.energy[term_id] for term_id in self.calc_energy["real"]]) #sum all energy terms relenat to group_id
    proposed_energy_terms = dict([(term_id, self.calc_energy["real"][term_id](proposed_real_params, self.complex_params)) for term_id in self.calc_energy["real"]])
    proposed_energy_partial = sum(proposed_energy_terms.values())
    accept = self.metropolis_decision(energy_partial, proposed_energy_partial)
    if accept:
      for term_id in self.calc_energy["real"]:
      	self.energy[term_id] = proposed_energy_terms[term_id] #update the relevant terms 
      self.real_params = proposed_real_params
    self.update_real_group_sigma(accept)
    return accept

  def step_all(self):
    """
    Step all parameters (amplitude, field coeffs) simultaneously. 
    Generic to any proposal distribution (draw function) and adaptive algorithm
    """
    proposed_real_params, proposed_complex_params = self.draw_real_group(), self.draw_complex_group()
    if self.reject_condition(proposed_real_params, proposed_complex_params):
      self.update_sigma(accept=False) 
      return False
    proposed_energy = self.calc_energy_total(proposed_real_params, proposed_complex_params)
    #print("energy", energy, "proposed_energy", proposed_energy)
    accept = self.metropolis_decision(self.energy_total, proposed_energy)
    if accept:
      #print("accepted", proposed_state, proposed_energy)
      self.energy_total = proposed_energy # TODO : saved at energy_dict because I dont expect a switch of method within a simulation
      self.real_params = proposed_real_params
      self.complex_params = proposed_complex_params
    self.update_sigma(accept)
    return accept

  def draw_real_group(self):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    """
    # no arguments - look at self.real_params
    #print(self.real_params, self.real_group_sampling_width, self.covariance_matrix_real)
    proposed_state = np.random.multivariate_normal(self.real_params,
                                                      self.real_group_sampling_width ** 2 * self.covariance_matrix_real,
                                                      check_valid='raise')
    #print(proposed_state)
    return proposed_state

  def draw_complex_group(self):
    """
    Get a random location in parameter space, from gaussian distribution around current location

    this version: draws from complex multivariate gaussian distribution

    Mapping to 2nx2n real-valued multivariate gaussian distribution and back, until complex multivariate is implemented in numpy 
    fully correct (for circular symmetric random complex variabels, zero pseudocovariance): uses all info in covariance matrix
    but involved decomposing and reassembling to real and img parts at each step - might be slow.

    While techinically correct, phase-varying generates greater amounts of statistically independent simulation result
    for situations where phases of the complex variables are weakly correlated with each other
    """
    #take complex vector & decompose to [reals, imgs]
    real_img_mean = np.concatenate((self.complex_params.real,self.complex_params.imag))
    #take complex covariance matrix
    #extract correlations Kxx, Kxy, Kyx, Kyy
    #between real parts(x) and img parts(y)
    Kxx=self.covariance_matrix_complex.real
    #Kxx=Kyy
    Kxy=self.covariance_matrix_complex.imag
    #Kyx=-Kxy
    #make real-valued covariance matrices [Kxx  Kxy 
    #                                     Kyx  Kyy] 
    real_img_cov = self.complex_group_sampling_width**2*0.5*np.block([[Kxx, Kxy],[-Kxy, Kxx]])
    #draw new real and img components in the form [reals, imgs]
    real_result, img_result = np.split(np.random.multivariate_normal(real_img_mean, real_img_cov),2)
    #compose new complex vector 
    return real_result+img_result*1j
  
  def draw_complex_magnitudes(self):
    covariances = (self.complex_group_sampling_width**2)*np.diagonal(self.covariance_matrix_complex)
    return np.array([self.modify_magnitude(mean, cov) for mean, cov in zip(self.complex_params, covariances)])

  def modify_magnitude(self, mean, cov):
    magnitude, phase = cmath.polar(mean)
    return cmath.rect(random.gauss(magnitude, cov), phase)

  def draw_complex_phases(self):
    return np.array([self.modify_phase(mean) for mean in self.complex_params])

  def modify_phase(self, mean):
    magnitude, phase = cmath.polar(mean)
    return cmath.rect(magnitude, random.uniform(-math.pi, math.pi))

  def metropolis_decision(self, old_energy, proposed_energy):
    """
    Considering energy difference and temperature, return decision to accept or reject step
    :param old_energy: current system total energy
    :param proposed_energy: system total energy after proposed change
    :return: bool True if step will be accepted; False to reject
    :rtype: bool
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
    #append to time series
    self.real_params_time_series.append(self.real_params)
    self.complex_params_time_series.append(self.complex_params)
    self.observables_time_series.append(self.observables)
    for term in self.energy:
    	self.energy_time_series[term].append(self.energy[term])
    self.real_group_sampling_width_time_series.append(self.real_group_sampling_width)
    self.complex_group_sampling_width_time_series.append(self.complex_group_sampling_width)

  #only used when parameter space is pure real
  def measure_real_system(self):
    self.measure_step_counter +=1
    self.measure_real()
    # other parameters that are not the basis for cov matrices:
    self.construct_observables()
    self.update_observables_mean()
    #append to time series
    self.real_params_time_series.append(self.real_params)
    self.observables_time_series.append(self.observables)
    for term in self.energy:
    	self.energy_time_series[term].append(self.energy[term])
    self.real_group_sampling_width_time_series.append(self.real_group_sampling_width)

  #only used when parameter space is pure complex. 
  def measure_complex_system(self):
    self.measure_step_counter +=1
    self.measure_complex()
    # other parameters that are not the basis for cov matrices:
    self.construct_observables()
    self.update_observables_mean()
    self.complex_params_time_series.append(self.complex_params)
    self.observables_time_series.append(self.observables)
    for term in self.energy:
    	self.energy_time_series[term].append(self.energy[term])
    self.complex_group_sampling_width_time_series.append(self.complex_group_sampling_width)

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
    self.covariance_matrix_complex *= (i - 2) / (i - 1)
    self.covariance_matrix_complex += np.outer(old_mean_complex, old_mean_complex.conjugate()) - i / (i - 1) * np.outer(self.complex_mean, self.complex_mean.conjugate()) +np.outer(self.complex_params, self.complex_params.conjugate()) / (i - 1) + small_number*np.identity(self.num_complex_params, dtype = 'complex128')

  def update_sigma(self, accept):
    step_number_factor = max((self.measure_step_counter / self.m, 200))
    steplength_c = self.sampling_width * self.ratio
    if accept:
      self.sampling_width += steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.sampling_width -= steplength_c * self.target_acceptance / step_number_factor
    self.real_group_sampling_width = self.sampling_width
    self.complex_group_sampling_width = self.sampling_width
    assert (self.sampling_width) > 0
  
  def update_real_group_sigma(self,accept):
    step_number_factor = max((self.measure_step_counter / self.m, 200))
    steplength_c = self.real_group_sampling_width * self.ratio
    if accept:
      self.real_group_sampling_width += steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.real_group_sampling_width -= steplength_c * self.target_acceptance / step_number_factor

  
  def update_complex_group_sigma(self,accept):
    self.step_counter +=1 # only count steps  while sigma was updated?
    step_number_factor = max((self.measure_step_counter / self.m, 200))
    steplength_c = self.complex_group_sampling_width * self.ratio
    if accept:
      self.complex_group_sampling_width += steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.complex_group_sampling_width -= steplength_c * self.target_acceptance / step_number_factor
 
  def construct_observables(self):
    # TODO : faster method?
    observables = [abs(x) for x in self.real_params]
    observables.extend([abs(x) for x in self.complex_params])
    observables.extend([x**2 for x in self.real_params])
    self.observables = np.array(observables)

  ############## collect, save data #####################
  def save_time_series(self):
    time_series_dict = dict([(name,[l[i] for l in  self.observables_time_series]) for i,name in enumerate(self.observables_names)])
    for name in self.energy:
      time_series_dict[name+"_energy"] = self.energy_time_series[name]
    if self.real_params_time_series:
      for i in range(self.num_real_params):
        time_series_dict[self.params_names[i]] = [l[i] for l in self.real_params_time_series]
      time_series_dict["real_group_sampling_width"] = self.real_group_sampling_width_time_series
    if self.complex_params_time_series:
      for i in range(self.num_complex_params):
        time_series_dict[self.params_names[self.num_real_params+i]] =[l[i] for l in self.complex_params_time_series]
      time_series_dict["complex_group_sampling_width"] = self.complex_group_sampling_width_time_series
    self.df = pandas.DataFrame.from_dict(time_series_dict)
    print(self.df)

  def save_equilibrium_stats(self):
    if self.df is None:
      self.save_time_series()
    self.eq_points = metropolisengine.statistics.get_equilibration_points(self.df)
    print(self.eq_points)
    self.global_eq_point = max([t for (key, [t,e,n]) in zip(self.eq_points.keys(), self.eq_points.values()) if "sampling_width" not in key]) #choice to count data
    #while sampling width is still trending up/down
    self.equilibrated_means, self.eq_means_error = metropolisengine.statistics.get_equilibrated_means(self.df, cutoff = self.global_eq_point)
    print("global t_0", self.global_eq_point) #add this data to means dict
    self.equilibrated_means["global_cutoff"] = self.global_eq_point

  ##############functions for complex number handling #########################

  @staticmethod
  def random_phase_complex(self, r):
    """
    a random complex number with a random (uniform between -pi and pi) phase and exactly the given magnitude r
    :param r: magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation
    """
    phi = random.uniform(-math.pi, math.pi)
    return cmath.rect(r, phi)

  @staticmethod
  def gaussian_complex(sigma):
    """
    a random complex number with completely random phase and amplitude drawn from a gaussian distribution with given width sigma (default 1)
    used for random initial values
    :param sigma: width of gaussian distribution (centered around 0) of possible magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation 
    """
    amplitude = random.gauss(0, sigma)
    phase = random.uniform(0, 2 * math.pi)
    return cmath.rect(amplitude, phase)


