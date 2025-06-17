# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1] #encoder(inputs,attention_mask=inputs.ne(1))得到的是一个元组，第一个元素是last_hidden_state，第二个元素是pooler_output
        #outputs的shape是(2*bs,hidden_size)
        code_vec=outputs[:bs] #bs是batch_size,最终得到的维度是(bs,hidden_size)
        nl_vec=outputs[bs:]
        
        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        #print(scores)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
