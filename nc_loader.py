import xarray as xr
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

class gen:
    def __call__(self, fname):
        dsg = xr.open_dataset(fname.decode("utf-8"))
        dsp = xr.open_dataset("/home/lar116/project/ERA5_ECMWF/era5_prec_2018H1.nc")
        for t in dsg.time:
            yield (dsg['z'].sel(time=t).data[:720, :, :], dsp['tp'].sel(time=t).data[:720, :, None]*1000)

        dsg.close()
        dsp.close()


def ERA5Dataset(fnames, batch_size=4):

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    #ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32,tf.float32), (tf.TensorShape([720, 1440, 3]),tf.TensorShape([720, 1440, 1])), args=(fname,)), cycle_length=len(fnames), block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32,tf.float32), (tf.TensorShape([720, 1440, 3]),tf.TensorShape([720, 1440, 1])), args=(fname,)), cycle_length=len(fnames), block_length=1, num_parallel_calls=min(4,len(fnames)))
    ds = ds.shuffle(128, seed=0)

    return ds.batch(batch_size)

