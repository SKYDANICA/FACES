#import openai
import pandas as pd
from tqdm import tqdm
import random
import time
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import torch
import re
import os

from get_case import get_test_case, get_explain

random.seed(1234)

model_name_or_path = "meta-llama/Llama-3.2-3B"


def get_pad_token_id(model_name, tokenizer):
    pad_token_id = None
    model_name = model_name.lower()
    if 'starcoder' in model_name:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    elif 'polycoder' in model_name:
        pad_token_id = tokenizer.get_vocab()['<|padding|>']
    elif 'codegen' in model_name:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    elif 'qwen' in model_name:
        pad_token_id = tokenizer('<|endoftext|>').input_ids[0]
    elif 'llama' in model_name:
        pad_token_id = tokenizer('<|finetune_right_pad_id|>').input_ids[0]
    elif 'gpt' in model_name:
        pad_token_id = tokenizer('<|endoftext|>').input_ids[0]
    return pad_token_id

def comment(example_source1, example_target1, example_source2,example_target2, example_source3, example_target3,example_source4, example_target4,example_source5,example_target5, example_source6, example_target6, example_source7,example_target7, example_source8, example_target8,example_source9, example_target9,example_source10,example_target10, code, num):#
    
    spe_token_ = tokenizer.encode("<s>", return_tensors="pt").to(device)
    spe_token = tokenizer.encode("</s>", return_tensors="pt").to(device)
    prompt_str = prompt_str = "You are a code expert. Please generate a brief summary of the code. To assist you, we will provide k examples for reference. Additionally, we have included test cases; please simulate the function execution to understand its internal logic：\n"
    prompt =  prompt_str + example_source10 + "}<s>" + str(example_target10) + ".</s>\n\n" + \
              example_source9 + "}<s>" + str(example_target9) + ".</s>\n\n" + \
              example_source8 + "}<s>" + str(example_target8) + ".</s>\n\n" + \
              example_source7 + "}<s>" + str(example_target7) + ".</s>\n\n" + \
              example_source6 + "}<s>" + str(example_target6) + ".</s>\n\n" + \
              code
              
    prompt_token = tokenizer.encode(prompt, return_tensors="pt").to(device)
    inputs = prompt_token.to(device)
    inputs = torch.cat((inputs, spe_token_), dim=1)
    #print(tokenizer.decode(inputs[0]))
    answer_begin_idx = inputs.shape[1]
    inputs = model.generate(inputs, max_new_tokens=32, pad_token_id=pad_token_id)
    answer = inputs[0][answer_begin_idx:]
    answer = tokenizer.decode(answer)
    answer = answer.replace('\n', '').replace('\t', ' ')
    answer = re.sub(r"\s+", " ", answer)# remove extra spaces
    if '</s>' in answer:
        answer = answer[:answer.index('</s>')]
        if '.' not in answer:
            answer = answer + "."
    if '.' in answer:
        answer = answer[:answer.index('.')] + '.'

    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_num", default=1, type=int, help="The num of test case.")
    args = parser.parse_args()
    path = "data/code_Retri.csv.csv"
    df = pd.read_csv(path)
    example_code1 = df['code1'].tolist()
    example_comment1 = df['comment1'].tolist()
    example_code2 = df['code2'].tolist()
    example_comment2 = df['comment2'].tolist()
    example_code3 = df['code3'].tolist()
    example_comment3 = df['comment3'].tolist()
    example_code4 = df['code4'].tolist()
    example_comment4 = df['comment4'].tolist()
    example_code5 = df['code5'].tolist()
    example_comment5 = df['comment5'].tolist()
    example_code6 = df['code6'].tolist()
    example_comment6 = df['comment6'].tolist()
    example_code7 = df['code7'].tolist()
    example_comment7 = df['comment7'].tolist()
    example_code8 = df['code8'].tolist()
    example_comment8 = df['comment8'].tolist()
    example_code9 = df['code9'].tolist()
    example_comment9 = df['comment9'].tolist()
    example_code10 = df['code10'].tolist()
    example_comment10 = df['comment10'].tolist()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    pad_token_id = get_pad_token_id(model_name_or_path, tokenizer)
    source_codes = []
    with open("../dataset/java/clean_test.jsonl",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            source_codes.append(js['clean_code'])
            #source_codes.append(" ".join(js['code_tokens']))
            
    path_explain = "../dataset/java/explain"
    path_case = "../dataset/java/test_case"
    train_case_examples = get_test_case(path_case, "train", len(source_codes), 10, args.test_num)
    test_case_examples = get_test_case(path_case, "test", len(source_codes), 10, args.test_num)
    train_explain = get_explain(path_explain, "train", len(source_codes), 10)
    test_explain = get_explain(path_explain, "test", len(source_codes), 10)
    num_batches = math.ceil(len(source_codes) / 100)
    all_ = []
    for batch_index in range(num_batches):
        start_index = batch_index * 100
        end_index = min(start_index + 100, len(source_codes))

        source_batch = source_codes[start_index:end_index]
        
        python_codes = []
        for i in tqdm(range(len(source_batch))):
            code_num = batch_index * 100 + i
            #print(example_batch[i])
            # python_codes.append(
            #     comment(example_code10[i], example_comment10[i], 
            #             example_code9[i], example_comment9[i], 
            #             example_code8[i], example_comment8[i], 
            #             example_code7[i], example_comment7[i], 
            #             example_code6[i], example_comment6[i],
            #             example_code5[i], example_comment5[i], 
            #             example_code4[i], example_comment4[i], 
            #             example_code3[i], example_comment3[i], 
            #             example_code2[i], example_comment2[i], 
            #             example_code1[i], example_comment1[i], 
            #             source_batch[i], i))
            python_codes.append(
                comment("Test case:\n" + train_case_examples[code_num][9] + "\n" + train_explain[code_num][9] + "\nCode:\n" + example_code10[i], 
                        example_comment10[i], 
                        "Test case:\n" + train_case_examples[code_num][8] + "\n" + train_explain[code_num][8] + "\nCode:\n" + example_code9[i], 
                        example_comment9[i], 
                        "Test case:\n" + train_case_examples[code_num][7] + "\n" + train_explain[code_num][7] + "\nCode:\n" + example_code8[i],
                        example_comment8[i], 
                        "Test case:\n" + train_case_examples[code_num][6] + "\n" + train_explain[code_num][6] + "\nCode:\n" + example_code7[i], 
                        example_comment7[i], 
                        "Test case:\n" + train_case_examples[code_num][5] + "\n" + train_explain[code_num][5] + "\nCode:\n" + example_code6[i],
                        example_comment6[i],
                        "Test case:\n" + train_case_examples[code_num][4] + "\n" + train_explain[code_num][4] + "\nCode:\n" + example_code5[i], 
                        example_comment5[i], 
                        "Test case:\n" + train_case_examples[code_num][3] + "\n" + train_explain[code_num][3] + "\nCode:\n" + example_code4[i], 
                        example_comment4[i], 
                        "Test case:\n" + train_case_examples[code_num][2] + "\n" + train_explain[code_num][2] + "\nCode:\n" + example_code3[i],
                        example_comment3[i], 
                        "Test case:\n" + train_case_examples[code_num][1] + "\n" + train_explain[code_num][1] + "\nCode:\n" + example_code2[i], 
                        example_comment2[i], 
                        "Test case:\n" + train_case_examples[code_num][0] + "\n" + train_explain[code_num][0] + "\nCode:\n" + example_code1[i],
                        example_comment1[i], 
                        "Test case:\n" + test_case_examples[code_num] + "\n" + test_explain[code_num] + "\nCode:\n" + source_batch[i], 
                        i))
            # python_codes.append(
            #     comment(train_explain[code_num][9] + "\n" + "\nCode:\n" + example_code10[i], 
            #             example_comment10[i], 
            #             train_explain[code_num][8] + "\n" + "\nCode:\n" + example_code9[i], 
            #             example_comment9[i], 
            #             train_explain[code_num][7] + "\n" + "\nCode:\n" + example_code8[i],
            #             example_comment8[i], 
            #             train_explain[code_num][6] + "\n" + "\nCode:\n" + example_code7[i], 
            #             example_comment7[i], 
            #             train_explain[code_num][5] + "\n" + "\nCode:\n" + example_code6[i],
            #             example_comment6[i],
            #             train_explain[code_num][4] + "\n" + "\nCode:\n" + example_code5[i], 
            #             example_comment5[i], 
            #             train_explain[code_num][3] + "\n" + "\nCode:\n" + example_code4[i], 
            #             example_comment4[i], 
            #             train_explain[code_num][2] + "\n" + "\nCode:\n" + example_code3[i],
            #             example_comment3[i], 
            #             train_explain[code_num][1] + "\n" + "\nCode:\n" + example_code2[i], 
            #             example_comment2[i], 
            #             train_explain[code_num][0] + "\n" + "\nCode:\n" + example_code1[i],
            #             example_comment1[i], 
            #             test_explain[code_num] + "\n" + "\nCode:\n" + source_batch[i], 
            #             i))
        all_.append(python_codes)
        #break
    with open(os.path.join("../", "Llama3B" + str(args.test_num) + ".output"), 'w', encoding='utf-8') as f:
        idx = 0
        for gold in all_:
            for line in gold:
                f.write(str(idx) + '\t' + line + '\n')
                
                idx += 1