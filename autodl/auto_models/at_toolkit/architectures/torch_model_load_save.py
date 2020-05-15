#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from urllib.parse import urlparse
from torch.utils import model_zoo


def load_from_url_or_local(url, model_dir):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        sd = torch.load(filepath)
    else:
        sd = model_zoo.load_url(url=url, model_dir=model_dir)
    return sd
