import os
import tqdm
import javalang
import csv
import ast

from oast_builder import build_sequence_oast, save_type_dict
from ast_builder import get_sequence_ast
import time

def parse_ast(java_code, level):
    programtokens = javalang.tokenizer.tokenize(java_code)
    if level == "file_level":
        programast = javalang.parser.parse(programtokens)
    elif level == "function_level":
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()
    return programast

## build DAST sequence
def build_DAST(programast):
    cache_dict = {"in_BOP": False}
    all_ast_tokens = get_sequence_ast(programast, cache_dict)
    #print(all_ast_tokens)
    return all_ast_tokens

## build SAST sequence
def build_SAST(programast):
    cache_dict = {
        "in_BOP": False,
        "symbol_table": {},
        "common_func": common_func,
        "common_type": common_type
    }
    all_oast_tokens = build_sequence_oast(programast, cache_dict)
    # print(all_ast_tokens)
    return all_oast_tokens

ONLY_OAST = False
tic = time.time()
csv_path = "origindata/GCJ_with_AST+OAST.csv"
ast_dict_path = "origindata/GCJ_AST_dictionary.txt"
oast_dict_path = "origindata/GCJ_OAST_dictionary.txt"
oast_common_func_path = "origindata/GCJ_OAST_func_list.txt"
oast_common_type_path = "origindata/GCJ_OAST_type_list.txt"
ast_token_dict = set()
oast_token_dict = set()
ast_token_dict_amount = {}
oast_token_dict_amount = {}
amount = 0
totol_len = 0
directory_path = 'origindata/googlejam4_src'
#file_level function_level
data_level = 'file_level'
all_csv_datas = []

with open(oast_common_type_path, 'r') as f:
    content = f.read()
    common_type = ast.literal_eval(content)
    print(f"common type: {len(common_type)}")
with open(oast_common_func_path, 'r') as f:
    content = f.read()
    common_func = ast.literal_eval(content)
    print(f"common func: {len(common_func)}")

for root, dirs, files in os.walk(directory_path):
    print(f"Parsing Folder {root}")
    for file in files:
        qindex = root.split("\\")[1]
        file_name = file
        code_path = os.path.join(root, file)
        with open(code_path, 'r', encoding="utf-8") as file:
            java_code = file.read()
            java_ast = parse_ast(java_code, level=data_level)

            all_oast_tokens = build_OAST(java_ast)
            for tok in all_oast_tokens:
                if tok not in ['[', ']']:
                    oast_token_dict.add(tok)
                    if tok not in oast_token_dict_amount:
                        oast_token_dict_amount[tok] = 0
                    oast_token_dict_amount[tok] += 1

            all_ast_tokens = all_oast_tokens
            if not ONLY_OAST:
                all_ast_tokens = build_AST(java_ast)
                for tok in all_ast_tokens:
                    if tok not in ['[',']']:
                        ast_token_dict.add(tok)
                        if tok not in ast_token_dict_amount:
                            ast_token_dict_amount[tok] = 0
                        ast_token_dict_amount[tok] += 1

            one_data = {
                "index": qindex,
                "file": file_name,
                "code": java_code.replace('\0', ''),
                "AST": " ".join(all_ast_tokens),
                "OAST": " ".join(all_oast_tokens),
            }
            all_csv_datas.append(one_data)

            amount += 1
            #print(len(all_ast_tokens))

print("\nOAST(and AST)generate using %.6f seconds" % (time.time() - tic))

print("Start Saving Datas")
#save_type_dict()
with open(ast_dict_path, "w", encoding="utf-8") as file:
    file.write("\n".join(ast_token_dict))
with open(oast_dict_path, "w", encoding="utf-8") as file:
    file.write("\n".join(oast_token_dict))

sorted_dict = dict(sorted(ast_token_dict_amount.items(), key=lambda item: item[1], reverse=True))
with open(ast_dict_path.replace("dictionary", "count"), "w", encoding="utf-8") as file:
    for key, value in sorted_dict.items():
        file.write(f"{key}: {value}\n")

sorted_dict = dict(sorted(oast_token_dict_amount.items(), key=lambda item: item[1], reverse=True))
with open(oast_dict_path.replace("dictionary", "count"), "w", encoding="utf-8") as file:
    for key, value in sorted_dict.items():
        file.write(f"{key}: {value}\n")

with open(csv_path, 'w', newline='', encoding="utf-8") as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=all_csv_datas[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(all_csv_datas)





