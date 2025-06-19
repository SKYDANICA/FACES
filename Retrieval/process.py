import pandas as pd

codes = []
nls = []
df = pd.read_csv('code-10.csv')
print(df)
for index, row in df.iterrows():
    for col_index, value in enumerate(row):
        codes.append(value[:-1] + " ")
df1 = pd.read_csv('sim-10.csv', encoding='utf-8')
for index, row in df1.iterrows():
    for col_index, value in enumerate(row):
        if "." not in value:
            value = value[:-1] + ".\n"
        nls.append(value[:-1] + " ")

all_examples = []
idx_code = 0
idx_nls = 0
for i in range(len(nls) // 10):
    pair = {}
    for k in range(10):
        pair[nls[idx_nls]] = codes[idx_code]
        idx_code += 1
        idx_nls += 1
    sorted_dict = pair
    temp_code = []
    temp_nl = []
    for key, value in sorted_dict.items():
        temp_nl.append(key)
        temp_code.append(value)
    all_examples.append(temp_code + temp_nl)

df = pd.DataFrame(all_examples,index = None,columns = ["code1","code2","code3","code4","code5","code6","code7","code8","code9","code10","comment1",'comment2',"comment3","comment4","comment5","comment6",'comment7',"comment8","comment9","comment10"])

df.to_csv('data/code_Retri.csv')
print("finish!!!")