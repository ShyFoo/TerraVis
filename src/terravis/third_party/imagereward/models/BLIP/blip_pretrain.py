'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Adapted from BLIP (https://github.com/salesforce/BLIP) by way of ImageReward (https://github.com/zai-org/ImageReward)
'''

import transformers
transformers.logging.set_verbosity_error()

from torch import nn
import os
from .med import BertConfig, BertModel
from .blip import create_vit, init_tokenizer

class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = "med_config.json",  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,
                 queue_size = 57600,
                 momentum = 0.995,
                 cache_dir = None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            cache_dir (str): download cache for the BERT tokenizer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.tokenizer = init_tokenizer(cache_dir=cache_dir)
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

