import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
from tqdm import tqdm

MODEL_NAME = "microsoft/unixcoder-base" #Pay attention to using the same tokenizer as the one used during the retriever training.

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 256
MAX_LENGTH = 256

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    tokenizer = RobertaTokenizer.from_pretrained(name)
    # model = RobertaModel.from_pretrained(name)
    path = "../Code_to_Text/checkpoint-best-mrr/model_name.pth"  #Replace the path
    model = torch.load(path)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        #for sent in sents:
        for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)
            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling {}".format(POOLING))
            # output_hidden_state [batch_size, hidden_size]
            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def compute_kernel_bias(vecs, n_components):
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main():
    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")
    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))
    code_list = []
    with open("../dataset/java/clean_test.jsonl",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            #code_list.append(" ".join(js["code_tokens"]).replace('\n', ' '))
            code_list.append(" ".join(js["docstring_tokens"]).replace('\n', ' '))
    vecs_func_body = sents_to_vecs(code_list, tokenizer, model)
    if USE_WHITENING:
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            vecs_func_body
        ], n_components=N_COMPONENTS)
        vecs_func_body = transform_and_normalize(vecs_func_body, kernel, bias) # [code_list_size, dim]
    else:
        vecs_func_body = normalize(vecs_func_body)# [code_list_size, 768]
    print(vecs_func_body.shape)
    import pickle
    f = open('model/code_vector_whitening.pkl', 'wb')
    pickle.dump(vecs_func_body, f)
    f.close()
    f = open('model/kernel.pkl', 'wb')
    pickle.dump(kernel, f)
    f.close()
    f = open('model/bias.pkl', 'wb')
    pickle.dump(bias, f)
    f.close()

if __name__ == "__main__":
    main()