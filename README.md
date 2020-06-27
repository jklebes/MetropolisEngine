# MetropolisEngine

status: in development for own use on the test case [cylinder](https://github.com/jklebes/cylinder);
      hosted at test.pypi

Markov chain Monte Carlo simulation on a physics system.  Mixed complex and real parameter space and adaptive. 
Works together with some external function that takes a list of (real and complex) parameters and returns an energy. 

INSTALL:

pip install --index-url https://test.pypi.org/simple metropolisengine

IMPORT:

import metropolisengine as me

USE:

see demo/

minimal example:
initialize MetropolisEngine object.  You must give at least an energy function ( `[real params] [complex params] -> float` ) and at least a list of one value as either initial_real_params or initial_complex_params.  Also set a non-zero temperature of 1k_bT.

```
def energy_function(x):
  return x**2
  
my_engine = me.MetropolisEngine(lambda real_params complex_params = energy_function(*real_params), initial_real_params=[0.0], temp=1)
```

run the simulation by looping over `step_all()`, and record data at every step with `measure()`

```
n_steps = 1000
for i in range(n_steps):
  my_engine.step_all()
  my_engine.measure()
```

look at measured quatities such as mean, (co-)variances, other default measured quantities.

```
print(my_engine.real_mean())
print(my_engine.covariance_matrix())
print(zip(my_engine.observables_names, my_engine.observables_mean))
```

finally prompt the finalization of time series data into a table and look at it

```
my_engine.save_time_series()
print(my_engine.df)
```
