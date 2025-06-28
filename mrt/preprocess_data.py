import csv
import ast
import os
import random
from collections import deque
from dataset import YDataset
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []
        self.ind = 0

    def set_children(self, children):
        self.children = children

    def preorder(self, depth, cutd):
        if cutd > 0 and depth >= cutd:
            return ""
        output = self.val + " "
        output_depth = f"{depth}" + " "
        if self.children is None:
            return output, output_depth
        for child in self.children:
            mid_o, mid_d = child.preorder(depth + 1, cutd)
            output += mid_o
            output_depth += mid_d
        return output, output_depth

    def postorder(self, depth, cutd):
        if cutd > 0 and depth >= cutd:
            return ""
        output = ''
        output_depth = ''
        if self.children is None:
            return output, output_depth
        for child in self.children:
            mid_o, mid_d = child.postorder(depth + 1, cutd)
            output += mid_o
            output_depth += mid_d
        output += self.val + " "
        output_depth += f"{depth}" + " "
        return output, output_depth

    def preorder_indices_values(self, indices, values, ind):
        lind = ind
        self.ind = lind
        if self.children is None:
            return self.ind, lind
        for child in self.children:
            cind, lind = child.preorder_indices_values(indices, values, lind + 1)
            indices[0].append(self.ind)
            indices[1].append(cind)
            values.append(1)
        return self.ind, lind

    def display(self, depth):
        print("-"*depth, self.val)
        cdepth = depth + 1
        for child in self.children:
            child.display(cdepth)

def BFS_ind(root):
    if not root:
        return
    queue = deque([root])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.ind)
        if node.children is not None:
            for child in node.children:
                queue.append(child)
    return result

def BFS_value(root):
    if not root:
        return
    queue = deque([root])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.children is not None:
            for child in node.children:
                queue.append(child)
    return result

def DFS_value(root):
    if not root:
        return
    queue = deque([root])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.children is not None:
            for child in node.children:
                queue.append(child)
    return result

def DFS_ind(root):
    if not root:
        return
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.ind)
        if node.children is not None:
            for child in reversed(node.children):
                stack.append(child)
    return result

def buildTree(arr):
    if not arr:
        return None
    i = 0
    childrens = []
    while i < len(arr):
        if arr[i] == "[":
            j = i + 1
            cnt = 1
            while cnt > 0:
                if arr[j] == "[":
                    cnt += 1
                elif arr[j] == "]":
                    cnt -= 1
                j += 1
            root.children = buildTree(arr[i + 1:j - 1])
            i = j
        else:
            root = TreeNode(arr[i])
            childrens.append(root)
            i += 1
    return childrens

def geth(tlist):
    h = 1
    max_h = 0
    for tok in tlist:
        if tok == '[':
            h += 1
            max_h = max(h, max_h)
        elif tok == ']':
            h -= 1
    return max_h

PIC_MAX = 512
d_amount = np.zeros(PIC_MAX + 1)
def preprocess_tree(csvfile, ast_type, cache_path, dictionary_data, cut_len):
    processed_data = {}
    cache_data = []
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # 跳过表头
    totalcount = 0
    totalline = 0
    curr_totalh = 0
    curr_totallen = 0
    curr_minlen = 100000
    curr_maxlen = 0
    curr_minh = 100000
    curr_maxh = 0
    global_tdict = {}
    print("---------------------------------------")
    print(f"Start Processed Data")
    print("---------------------------------------")
    for row in csv_reader:
        totalcount += 1
        if totalcount % 1000 == 0:
            print(f"Processed Amount {totalcount}")
        index = row[0]
        file = row[1]
        code = row[2]

        code_line = [s.strip() for s in code.split("\n") if s]
        code_line = len(code_line)

        totalline += code_line
        ast1 = row[3]
        ast2 = row[4]
        if len(row) != 5:
            print(len(row))
        if ast_type == "SAST" or ast_type == "AST":
            tree_seq = ast1
        else:
            tree_seq = ast2

        token_list = [s.strip() for s in tree_seq.split(" ") if (s.strip() and s.strip() != '[' and s.strip() != ']')]
        token_tree_list = [s.strip() for s in tree_seq.split(" ") if (s.strip())]
        tree_h = geth(token_tree_list)
        curr_totalh = curr_totalh + tree_h
        tlen = len(token_list)
        curr_totallen = curr_totallen + tlen
        if tlen < curr_minlen:
            curr_minlen = tlen
        if tlen > curr_maxlen:
            curr_maxlen = tlen
        if tree_h < curr_minh:
            curr_minh = tree_h
        if tree_h > curr_maxh:
            curr_maxh = tree_h
        if tlen > PIC_MAX:
            d_amount[PIC_MAX] = d_amount[PIC_MAX] + 1
        else:
            d_amount[tlen] = d_amount[tlen] + 1
        tdict = {}
        for token in token_list:
            if token not in tdict:
                tdict[token] = 1
            else:
                tdict[token] += 1
            if token not in global_tdict:
                global_tdict[token] = 1
            else:
                global_tdict[token] += 1
        #print(file)
        try:
            tree_root = buildTree(token_tree_list)
        except Exception as e:
            print(f"ERROR AT : {index}-{file}")
        # dfs_value = DFS_value(tree_root[0])
        # bfs_value = BFS_value(tree_root[0])
        # print(dfs_value, bfs_value)

        preorder_seq, preorder_d = tree_root[0].preorder(0, 0)
        postorder_seq, postorder_d = tree_root[0].postorder(0, 0)
        preorder_seq_list = [item for item in preorder_seq.split(" ") if item]
        postorder_seq_list = [item for item in postorder_seq.split(" ") if item]
        preorder_d_list = [int(item) for item in preorder_d.split(" ") if item]
        postorder_d_list = [int(item) for item in postorder_d.split(" ") if item]
        current_len = len(preorder_seq_list)

        # cut_h = tree_h
        # while current_len > cut_len:
        #     preorder_seq = tree_root[0].preorder(0, cut_h)
        #     postorder_seq = tree_root[0].postorder(0, cut_h)
        #     preorder_seq_list = [item for item in preorder_seq.split(" ") if item]
        #     postorder_seq_list = [item for item in postorder_seq.split(" ") if item]
        #     current_len = len(preorder_seq_list)
        #     cut_h -= 1
        #     #print(f"CUT h={cut_h} len={current_len}")

        preorder_seq_id = [dictionary_data[s] for s in preorder_seq_list]
        postorder_seq_id = [dictionary_data[s] for s in postorder_seq_list]
        values = []
        indices = [[], []]
        tree_root[0].preorder_indices_values(indices, values, 0)
        bfs_ind = BFS_ind(tree_root[0])
        dfs_ind = DFS_ind(tree_root[0])

        if index not in processed_data:
            processed_data[index] = {}
        processed_data[index][file] = {
            "code": code,
            "tree_seq": tree_seq,
            "preorder": preorder_seq_id,
            "postorder": postorder_seq_id,
            "preorder_d": preorder_d_list,
            "postorder_d": postorder_d_list,
            "values": values,
            "indices": indices,
            "bfs_ind": bfs_ind,
            "dfs_ind": dfs_ind,
            "len": tlen,
            "tdict":tdict
        }
        cache_data.append({
            "index": index,
            "file": file,
            "preorder": preorder_seq_id,
            "postorder": postorder_seq_id,
            "values": values,
            "indices": indices,
            "bfs_ind": bfs_ind,
            "dfs_ind": dfs_ind,
            "len": tlen,
        })
    print("---------------------------------------")
    print("Finish Processed Data, Total: ", totalcount, " avg code line: ", totalline / totalcount)
    print("max len: ", curr_maxlen, " min len: ", curr_minlen, " avg len: ", curr_totallen / totalcount)
    print("max depth: ", curr_maxh, " min depth: ", curr_minh, "avg depth: ", curr_totalh / totalcount)
    if "SelectionStatement" in global_tdict:
        print("avg selection nodes: ", global_tdict["SelectionStatement"] / totalcount)
    if "LoopStatement" in global_tdict:
        print("avg loop nodes: ",global_tdict["LoopStatement"] / totalcount)
    if "FixedSelectionStmtClass" in global_tdict:
        print("avg selection nodes: ", global_tdict["FixedSelectionStmtClass"] / totalcount)
    if "FixedLoopStmtClass" in global_tdict:
        print("avg loop nodes: ",global_tdict["FixedLoopStmtClass"] / totalcount)

    if "IfStatement" in global_tdict:
        print("avg if nodes: ", global_tdict["IfStatement"] / totalcount)
    if "SwitchStatement" in global_tdict:
        print("avg switch nodes: ",global_tdict["SwitchStatement"] / totalcount)
    if "ForStatement" in global_tdict:
        print("avg for nodes: ", global_tdict["ForStatement"] / totalcount)
    if "WhileStatement" in global_tdict:
        print("avg while nodes: ",global_tdict["WhileStatement"] / totalcount)
    if "DoStatement" in global_tdict:
        print("avg do-while nodes: ",global_tdict["DoStatement"] / totalcount)
    print("---------------------------------------")
    return processed_data

def draw_len_pic():
    d_len = np.arange(1, PIC_MAX + 2, 1)
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.plot(d_len, d_amount)
    plt.xlabel('len')
    plt.ylabel('amount')
    plt.title('total')
    plt.savefig(f'analyze_pic/total.png')

def process_pair(pair_file):
    csv_reader = csv.reader(pair_file)
    next(csv_reader)
    amount = 0
    pair_datas = []
    for row in csv_reader:
        amount += 1
        index1 = row[0]
        file1 = row[1]
        index2 = row[2]
        file2 = row[3]
        label = int(row[4])
        ctype = "None"
        if label == -1:
            label = 0
        pair_data = {
            "index1": index1,
            "index2": index2,
            "file1": file1,
            "file2": file2,
            "label": label,
            "type": ctype,
        }
        pair_datas.append(pair_data)
    print(f"Process -> {pair_file.name} Pair Amount: {amount}")
    return pair_datas

all_appear_tdict = []
def analyze_pair(pair_data, processed_data, cut_len):
    amount = [0, 0]
    amount_out = 0
    appear_tdict = {}
    for row in pair_data:
        index1 = row["index1"]
        file1 = row["file1"]
        index2 = row["index2"]
        file2 = row["file2"]
        data1 = processed_data[index1][file1]
        data2 = processed_data[index2][file2]
        len1 = len(data1["preorder"])
        len2 = len(data2["preorder"])
        tdict1 = data1["tdict"]
        tdict2 = data1["tdict"]
        for key, value in tdict1.items():
            if key not in appear_tdict:
                appear_tdict[key] = 1
            else:
                appear_tdict[key] += 1
        for key, value in tdict2.items():
            if key not in appear_tdict:
                appear_tdict[key] = 1
            else:
                appear_tdict[key] += 1

        if len1 < 8 or len1 > cut_len or len2 < 8 or len2 > cut_len:
            amount_out += 1
        label = int(row["label"])
        if label == -1 or label == 0:
            amount[0] += 1
        if label == 1:
            amount[1] += 1

    all_appear_tdict.append(appear_tdict)
    rate_01 = amount[0] / amount[1]
    print(f"Analyze -> | Pair Amount[0,1]: {amount} {{{rate_01:.4f}:1}} | Out of [8,{cut_len}]: {amount_out}")

def read_from_cache(cache_file):
    csv_reader = csv.reader(cache_file)
    amount = 0
    processed_data = {}
    for row in csv_reader:
        amount += 1
        if amount % 1000 == 0:
            print(f"Load from Cache {amount}")
        index = row[0]
        file = row[1]
        preorder_seq = ast.literal_eval(row[2])
        postorder_seq = ast.literal_eval(row[3])
        values = ast.literal_eval(row[4])
        indices = ast.literal_eval(row[5])
        bfs_ind = ast.literal_eval(row[6])
        dfs_ind = ast.literal_eval(row[7])
        tlen = int(row[8])

        if index not in processed_data:
            processed_data[index] = {}
        processed_data[index][file] = {
            "preorder": preorder_seq,
            "postorder": postorder_seq,
            "values": values,
            "indices": indices,
            "bfs_ind": bfs_ind,
            "dfs_ind": dfs_ind,
            "len": tlen
        }
    print(f"Load from Cache Data Amount: {amount}")
    return processed_data

def get_dictionary(dic_file):
    token_dict = {}
    index = 1
    for line in dic_file:
        token_dict[line.strip()] = index
        index += 1
    print(f"Dict Size: {len(token_dict)}")
    return token_dict

def getdata(data_csv_path, train_path, valid_path, test_path, dictionary_path, ast_type, cut_len, quick_test = False):
    cache_path = data_csv_path.replace("origindata", "data")
    token_dictionary_path = dictionary_path
    token_dictionary_path = token_dictionary_path.replace("XXX", ast_type)
    dictionary_file = open(token_dictionary_path, "r", encoding="utf-8")
    dictionary_data = get_dictionary(dictionary_file)
    word_size = len(dictionary_data)

    csvfile = open(data_csv_path, "r", encoding="utf-8")
    processed_data = preprocess_tree(csvfile, ast_type, cache_path, dictionary_data, cut_len)

    # if os.path.exists(cache_path):
    #     csvfile = open(cache_path, "r", encoding="utf-8")
    #     processed_data = read_from_cache(csvfile)
    # else:
    #     csvfile = open(data_csv_path, "r", encoding="utf-8")
    #     processed_data = preprocess_tree(csvfile, ast_type, cache_path, dictionary_data)
    if quick_test:
        test_pair = open(test_path, "r", encoding="utf-8")
        test_pair_datas = process_pair(test_pair)
        test_set = YDataset(processed_data, test_pair_datas, True, cut_len)
        return test_set, word_size
    else:
        train_pair = open(train_path, "r", encoding="utf-8")
        train_pair_datas = process_pair(train_pair)
        analyze_pair(train_pair_datas, processed_data, cut_len)
        valid_pair = open(valid_path, "r", encoding="utf-8")
        valid_pair_datas = process_pair(valid_pair)
        analyze_pair(valid_pair_datas, processed_data, cut_len)
        test_pair = open(test_path, "r", encoding="utf-8")
        test_pair_datas = process_pair(test_pair)
        analyze_pair(test_pair_datas, processed_data, cut_len)

        training_set = YDataset(processed_data, train_pair_datas, True, cut_len)
        valid_set = YDataset(processed_data, valid_pair_datas, True, cut_len)
        test_set = YDataset(processed_data, test_pair_datas, True, cut_len)

        return [training_set, valid_set, test_set], word_size

def print_appear_tdict():
    sorted_dict = dict(sorted(all_appear_tdict[0].items(), key=lambda item: item[1], reverse=True))
    with open("analyze_pic/train_tdict.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            file.write(f"{key}: {value}\n")
    sorted_dict = dict(sorted(all_appear_tdict[1].items(), key=lambda item: item[1], reverse=True))
    with open("analyze_pic/valid_tdict.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            file.write(f"{key}: {value}\n")
    sorted_dict = dict(sorted(all_appear_tdict[2].items(), key=lambda item: item[1], reverse=True))
    with open("analyze_pic/test_tdict.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    data_csv_path = "origindata/GCJ_with_AST+OAST.csv"
    train_path = "origindata/GCJ_train11.csv"
    valid_path = "origindata/GCJ_valid.csv"
    test_path = "origindata/GCJ_test.csv"
    dictionary_path = "origindata/GCJ_XXX_dictionary.txt"
    [training_set, valid_set, test_set], word_size = getdata(data_csv_path, train_path, valid_path, test_path, dictionary_path, True, 256)

    draw_len_pic()
    print_appear_tdict()

