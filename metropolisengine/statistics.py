import pandas
import matplotlib.pyplot as plt
import numpy as np
from pymbar import timeseries


def plottable_timeseries_from_csv(in_dir, file_name, column_name=None):
  data = pandas.read_csv(file_name, index_col = 0)
  all_timeseries=dict([])
  if column_name is None:
    for name in data.columns:
      time_series = data[name]
      if not isinstance(time_series[0], float):
        timeseries_real = [complex(y).real for y in time_series]
        if complex(time_series[0]).imag!=0:
          timeseries_imag =  [complex(y).imag for y in time_series]
          all_timeseries[name+"_imag"] = timeseries_imag
        all_timeseries[name+"_real"] = timeseries_real
      else:
        all_timeseries[name]=time_series
  else:
    all_timeseries[name] = data[name]
  return all_timeseries

def get_equilibration_points(df, column_name = None):
  """
  directly uses pymbar's timeseries utility!
  source: https://github.com/choderalab/pymbar
  https://pymbar.readthedocs.io/en/master/timeseries.html 
  and references therein
  """
  equilibration_data=dict([])
  for name in df.columns.values:
    """
    t - t_0 starting point of equilibrated part of series
    g - the statistical innefficency = 2*correlationtime +1
    N_effmax - effective number of uncorrelated samples
    """
    if isinstance(df.loc[0, name] , complex):
      [t,g,Neff_max] = timeseries.detectEquilibration(np.array([x.real for x in df.loc[:,name]]))  
      equilibration_data[name+"_real"] = [t,g,Neff_max]
      [t,g,Neff_max] = timeseries.detectEquilibration(np.array([x.imag for x in df.loc[:,name]]))  
      equilibration_data[name+"_imag"] = [t,g,Neff_max]
    else:
      [t, g, Neff_max] = timeseries.detectEquilibration(df.loc[:,name]) 
      equilibration_data[name] = [t,g,Neff_max]
  print(equilibration_data)
  return equilibration_data

def cut_series(out_dir, file_name, column_name=None):
  pass

def get_equilibrated_means(df, cutoff = None):
  """ re-average time series from supplied starting point
  (or infer system global t0 equilibration point)
  """
  if cutoff is None:
    eq_data = self.find_equilibrium(all_timeseries)
    cutoff = max([t for [t,g,n] in eq_data])
  means  = dict([])
  errors = dict([])
  for name in df.columns.values:
    means[name] = np.average(df.loc[cutoff:, name])
  return means, errors

#function to plot
def plot_time_series(out_dir, file_name, column_name=None, log=True):
  data = pandas.read_csv(file_name, index_col = 0)
  print(data)
  ts = [1+i for i in data.index] #put time from 1 to n(incl), rather than 0 to n-1(incl), displays better on log time scale
  if column_name is None:
    # just plt everything
    data_sampling_width = dict([])
    data_abs = dict([])
    data_squared = dict([])
    data_real = dict([])
    data_realpart = dict([])
    data_imgpart = dict([])
    #raw values of complex coefficients excludes from plotting
    for name in data.columns:
      ys = data[name]
      if not isinstance(ys[0], float):
        if complex(ys[0]).imag==0:
          ys = [complex(y).real for y in ys]
        else:
          data_realpart[name]=[complex(y).real for y in ys]
          data_imgpart[name]=[complex(y).imag for y in ys]
          ys=None
      if ys is not None: #float series or extracted real series from (x+0j) complex string series
        if "abs" in name:
          data_abs[name]=ys
        elif "squared" in name:
          data_squared[name] = ys
        elif "sampling_width" in name:
          data_sampling_width[name] = ys
        else:
          data_real[name] = ys
    for name in data_abs:
      plt.scatter(ts,data_abs[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_abs.png")
    plt.close()
    for name in data_squared:
      plt.scatter(ts,data_squared[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_squared.png")
    plt.close()
    for name in data_sampling_width:
      plt.scatter(ts,data_sampling_width[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_sigma.png")
    plt.close()
    for name in data_real:
      plt.scatter(ts,data_real[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_realparams.png")
    plt.close()
    for name in data_realpart:
      plt.scatter(ts,data_realpart[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_realpart.png")
    plt.close()
    for name in data_imgpart:
      plt.scatter(ts,data_imgpart[name], label=name)
    if log:
      plt.xscale('log')
    plt.legend()
    plt.savefig("exampletimeseries_imgpart.png")
    plt.close()


if __name__=="__main__":
  #for post-hoc analysis 
  outdir = ".."
  filename = "../exampledata.csv"
  #filename = "ncoeffs(3, 0)_fsteps1.csv"
  plot_time_series(outdir, filename)
  find_equilibrium(outdir, filename)
  cut_series(outdir, filename)
  all_timeseries = timeseries_from_csv(out_dir, file_name, column_name)
