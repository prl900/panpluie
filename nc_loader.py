import xarray as xr
import tensorflow as tf
import datetime

tf.compat.v1.enable_eager_execution()

class gen:
    def __call__(self, fname):
        dsg = xr.open_dataset(fname.decode("utf-8"))
        dsp = xr.open_dataset("/data/ERA5/era5_prec_2018H1.nc")
        for t in dsg.time:
            #yield (dsg['z'].sel(time=t).data, dsp['tp'].sel(time=t).data)
            yield (dsg['z'].sel(time=t).data, t)


fnames = ["/data/ERA5/era5s_geop_201801.nc", 
          "/data/ERA5/era5s_geop_201802.nc", 
          "/data/ERA5/era5s_geop_201803.nc", 
          "/data/ERA5/era5s_geop_201804.nc", 
          "/data/ERA5/era5s_geop_201805.nc", 
          "/data/ERA5/era5s_geop_201806.nc"] 


ds = tf.data.Dataset.from_tensor_slices(fnames)
#ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32,tf.float32), (tf.TensorShape([721, 1440, 3]),tf.TensorShape([721, 1440])), args=(fname,)), cycle_length=1, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32,tf.int64), (tf.TensorShape([721, 1440, 3]),tf.TensorShape([])), args=(fname,)), cycle_length=2, block_length=2, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for value in ds.take(6):
    print(datetime.datetime.utcfromtimestamp(tf.cast(value[1], tf.float64) * 1e-9))
