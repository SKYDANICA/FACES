import os
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_test_case(path, type_data, num_data, shot_num, num_case):
    examples = []
    if type_data == "train_new":
        for i in range(num_data):
            example_one = []
            for j in range(shot_num):
                filename = str(i+1) + "_" + str(j+1) + ".txt"
                temp_path = os.path.join(path, type_data, filename)
                with open(temp_path, "r") as f:
                    res = f.read()
                    f.close()
                lines = res.split("\n")
                res = "\n".join(lines[:num_case])
                example_one.append(res)
            examples.append(example_one)
    elif type_data == "test_new":
        for i in range(num_data):
            temp_path = os.path.join(path, type_data, str(i+1)+".txt")
            with open(temp_path, "r") as f:
                res = f.read()
                f.close()
            lines = res.split("\n")
            res = "\n".join(lines[:num_case])
            examples.append(res)
    else:
        logger.warning("You selected the wrong type_data, only 'train' and 'test' can be selected.")
    return examples


def get_explain(path, type_data, num_data, shot_num):
    examples = []
    if shot_num < 0 or shot_num > 10:
        logger.warning("You chose the wrong shot_num.")
        return
    if type_data == "train_new":
        for i in range(num_data):
            example_one = []
            for j in range(shot_num):
                filename = str(i+1) + "_" + str(j+1) + ".txt"
                temp_path = os.path.join(path, type_data, filename)
                with open(temp_path, "r") as f:
                    res = f.read()
                    f.close()
                example_one.append(res)
            examples.append(example_one)
    elif type_data == "test_new":
        for i in range(num_data):
            temp_path = os.path.join(path, type_data, str(i+1)+".txt")
            with open(temp_path, "r") as f:
                res = f.read()
                f.close()
            examples.append(res)
    else:
        logger.warning("You selected the wrong type_data, only 'train' and 'test' can be selected.")
    return examples

if __name__ == '__main__':
    path = "../RAG/explain"
    res = get_explain(path, "train", 10, 3)
    print(len(res[0]))