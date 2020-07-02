import pandas
import matplotlib.pyplot as plt

#read time series dataframe from csvfile


#function to plot
def plot_time_series(out_dir, file_name, column_name=None):
  data = pandas.read_csv(file_name, index_col = 0)
  print(data)
  if column_name is None:
    x=[1,2]
    y=[3,2.1]
    plt.scatter(x,y)
    plt.savefig("exampletimeseries.png")
    plt.close()
  pass

#correlation time?

#stationarity

if __name__=="__main__":
  outdir = ".."
  filename = "../exampledata.csv"
  plot_time_series(outdir, filename)
