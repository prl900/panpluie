import tensorflow as tf
import os
import xarray as xr

class GenerateTFRecord:
    def __init__(self, year, month):
        self.dsg = xr.open_dataset("/data/ERA5/era5s_geop_{}{:02d}.nc".format(year, 1 + month//7))
        self.dsp = xr.open_dataset("/data/ERA5/era5_prec_{}H{}.nc".format(year, 1 + month//7))

    def convert(self, tfrecord_file_name):
        # Get all file names of images present in folder

        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            for t in self.dsg.time:
                x = self.dsg['z'].sel(time=t).data
                y = self.dsp['tp'].sel(time=t).data
                print(x.shape, y.shape, x.dtype, len(x.tostring()), len(y.tostring()))
                exit()
                sample = tf.train.Example(features = tf.train.Features(feature = {
                         #'timestamp': tf.train.Feature(bytes_list = tf.train.BytesList(value = [str(t.data)[:-10].encode('utf-8')])),
                         #'height': tf.train.Feature(int64_list = tf.train.Int64List(value = [x.shape[0]])),
                         #'width': tf.train.Feature(int64_list = tf.train.Int64List(value = [x.shape[1]])),
                         #'levels': tf.train.Feature(int64_list = tf.train.Int64List(value = [x.shape[2]])),
                         'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x.tostring()])),
                         'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [y.tostring()]))}))

                writer.write(sample.SerializeToString())

if __name__ == '__main__':
    t = GenerateTFRecord(2018, 1)
    t.convert('images.tfrecord')
