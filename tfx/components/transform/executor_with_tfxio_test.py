# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.transform.executor.

With the TFXIO code path being exercised.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.transform import executor_test


class ExecutorWithTFXIOTest(executor_test.ExecutorTest):

  def _use_tfxio(self):
    return True

  def _get_source_data_dir(self):
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')


if __name__ == '__main__':
  # TODO(b/150159972): remove once TFT post-0.21.0 released and depended on.
  if tft.__version__ > '0.21.0':
    tf.test.main()
