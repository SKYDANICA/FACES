import os
def get_info(path, mode, total_num, test_case_num, train_case_num=10):
    examples = []
    if mode == "test":
        for i in range(total_num):
            example_one = ""
            for j in range(test_case_num):
                file_name = str(i+1) + "_" + str(j+1) + ".txt"
                temp_path = os.path.join(path, mode, file_name)
                with open(temp_path, "r") as f:
                    res = f.read()
                    f.close()
                example_one = example_one + res + "\n"
            examples.append(example_one)
    elif mode == "train":
        for i in range(total_num):
            example = []
            for k in range(train_case_num):
                example_one = ""
                for j in range(test_case_num):
                    file_name = str(i+1) + "_" + str(k+1) + "_" + str(j+1) + ".txt"
                    temp_path = os.path.join(path, mode, file_name)
                    with open(temp_path, "r") as f:
                        res = f.read()
                        f.close()
                    example_one = example_one + res + "\n"
                example.append(example_one)
            examples.append(example)
    else:
        print("mode Error")
    return examples