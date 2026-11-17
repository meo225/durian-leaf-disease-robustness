# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Trích từ imagenet_c_bar/base.py (commit ba4d5a5). Chỉ giữ lớp Transform và đường gọi __call__, bỏ các hàm convert_* không dùng.

import abc


class Transform(abc.ABC):

    name = "abstract_transform"

    def __init__(self, severity, im_size, record=False, max_intensity=False, **kwargs):
        self.im_size = im_size
        self.severity = severity
        self.record = record
        self.max_intensity = max_intensity

    @abc.abstractmethod
    def transform(self, image, **kwargs):
        ...

    @abc.abstractmethod
    def sample_parameters(self):
        ...

    def __call__(self, image):
        params = self.sample_parameters()
        out = self.transform(image, **params)
        if self.record:
            return out, params
        return out
