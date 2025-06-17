# Function-Aware Code Summarization via Consistency Filtering and Test Case Augmentation
    This is the code for the paper "Function-Aware Code Summarization via Consistency Filtering and Test Case Augmentation." If you want to use it, you can refer to the following steps.

## Step-1 Download the dataset.
### Dataset
We use the Java and Python dataset from the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) code-to-text docstring
generation task, which is built upon the CodeSearchNet corpus and excludes defective data samples. 

We further process them to obtain two additional fields: 'clean_code' and 'clean_doc'.

#### Download data and preprocess

    unzip dataset.zip
    cd dataset
    wget https://zenodo.org/record/7857872/files/java.zip
    wget https://zenodo.org/record/7857872/files/python.zip
    
    unzip python.zip
    unzip java.zip

    python preprocess.py

    rm *.pkl
    rm -r */[^clean]*
    cd ..


#### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. clean_train.jsonl, clean_valid.jsonl, clean_test.jsonl

For each file, each line in the uncompressed file represents one function. Here is an explanation of the fields:

* The fields contained in the original CodeXGLUE dataset:

  * repo: the owner/repo

  * path: the full path to the original file

  * func_name: the function or method name

  * original_string: the raw string before tokenization or parsing

  * language: the programming language

  * code/function: the part of the original_string that is code

  * code_tokens/function_tokens: tokenized version of code

  * docstring: the top-level comment or docstring, if it exists in the original string

  * docstring_tokens: tokenized version of docstring

* The additional fields we added:

  * clean_code: clean version of code that removing possible comments

  * clean_doc: clean version of docstring that obtaining by concatenating docstring_tokens

#### Data Statistic

| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| Java                 | 164,923  | 5,183  | 10,955 |

## Step-2 Train the Code-to-Text and Code-to-Code retrievers.
    Here, please refer directly to the introductions in the Code-to-Text and Code-to-Code files, which will not be repeated here.

## Step-3 Retrieval Examples
    1、The specific methods of operation can be referred to in the introduction of the Retrieval file.
    2、It is important to note that the database should have been built first in this case. There is a slight deviation from the actual process, which is explained as follows: In our initial experiment using the Java dataset, the training set contained over 250,000 entries. If we had generated content for each data entry, we would have had to produce test case information for all 250,000 entries. However, the test set consists of only 10,000 entries. By retrieving five or eight examples for each test entry, we only need to generate test case information for 80,000 entries. Even with ablation experiments included, the total required information would be 240,000 entries. Thus, from a cost perspective, our experimental design differs slightly from the actual setup, but the final outcome remains unchanged.

## Step-4 Generate test case information.
    You can refer to the content in the Build_Database file. Note that if you are using a smaller-scale model to generate content, some of the generated data may not comply with our specifications. You may need to manually adjust these parts to facilitate the next steps. The likelihood of encountering such issues is reduced when using a larger-scale model.

## Step-5 Generate
    You can refer to the introduction in the Generate file.

