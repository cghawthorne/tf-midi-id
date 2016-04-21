"""Functions for downloading and reading MNIST data."""
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy
import tensorflow as tf

SAMPLE_DURATION_MILLIS = 10000
SAMPLE_WINDOW_MILLIS = 100
MAX_NOTE_VALUE = 100
INPUT_SIZE = int(SAMPLE_DURATION_MILLIS / SAMPLE_WINDOW_MILLIS * MAX_NOTE_VALUE)
SAMPLE_SIZE = int(SAMPLE_DURATION_MILLIS / SAMPLE_WINDOW_MILLIS)


COMPOSERS = {'bach': 0, 'children': 1}

def extract_midi_data(filename):
  midi_data = []
  with open(filename) as f:
    for line in f:
      fields = line.split(', ')
      if len(fields) < 5:
        continue
      if fields[2] != 'Note_on_c':
        continue
      # time and note value
      midi_data.append([float(fields[1]), int(fields[4])])
  midi_data = sorted(midi_data, key=lambda event: event[0])
  data = None
  # artifically increase sample data by shifting the data into different windows
  for x in xrange(SAMPLE_WINDOW_MILLIS):
    while midi_data[0][0] < x:
      midi_data.pop(0)
    shiftdata = numpy.zeros((int(math.ceil(midi_data[-1][0]/SAMPLE_WINDOW_MILLIS))+1, MAX_NOTE_VALUE), dtype=int)
    for event in midi_data:
      curtime = event[0]
      note = event[1]
      shiftdata[int(math.floor(curtime/SAMPLE_WINDOW_MILLIS)), note] += 1
    if data is None:
      data = shiftdata
    else:
      data = numpy.vstack((data, shiftdata))

  # make sure we have full 10-second samples
  sample_data_length = int(SAMPLE_DURATION_MILLIS/SAMPLE_WINDOW_MILLIS)
  windows = int(data.shape[0]/sample_data_length)*sample_data_length
  data = numpy.resize(data, (windows, data.shape[1]))

  print('Read %d windows from %s' % (windows, filename))
  
  return data.reshape((int(data.shape[0]/sample_data_length), int(data.shape[1]*sample_data_length)))


def extract_data(dirname):
  print('Extracting', dirname)
  labels = numpy.array([])
  data = None
  for (dirpath, dirnames, filenames) in os.walk(dirname):
    for filename in [f for f in filenames if f.endswith('.csv')]:
      labelval = COMPOSERS[os.path.split(dirpath)[-1]]
      #label = numpy.zeros(len(COMPOSERS), dtype=numpy.uint8)
      #label[labelval] = 1
      midi_data = extract_midi_data(os.path.join(dirpath, filename))
      if data is None:
        data = midi_data
      else:
        data = numpy.vstack((data, midi_data))
      labels = numpy.append(labels, [labelval]*midi_data.shape[0])
      #for x in range(0, midi_data.shape[0]):
      #  if labels is None:
      #    labels = label
      #  else:
      #    labels = numpy.vstack((labels, label))

  return DataSet(data, labels)


class DataSet(object):

  def __init__(self, midi_data, labels):
    assert midi_data.shape[0] == labels.shape[0], ('midi_data.shape: %s labels.shape: %s' % (midi_data.shape, labels.shape))
    self._num_examples = midi_data.shape[0]
    self._midi_data = midi_data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def midi_data(self):
    return self._midi_data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._midi_data = self._midi_data[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._midi_data[start:end], self._labels[start:end]


def read_data_sets(train_dir):
  class DataSets(object):
    pass
  data_sets = DataSets()

  data_sets.train = extract_data(os.path.join(train_dir, 'train'))
  data_sets.validation = extract_data(os.path.join(train_dir, 'validation'))
  data_sets.test = extract_data(os.path.join(train_dir, 'test'))

  return data_sets


def load_midis():
    return read_data_sets("midis")

