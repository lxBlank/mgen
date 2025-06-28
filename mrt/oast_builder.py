from javalang.ast import Node
import ast
import copy

TO_CLEAN = True
TO_ENHANCE = True
TO_UNIFY = True

remove_node = ["LocalVariableDeclaration","Literal","VoidClassReference","ReferenceType","ClassReference","BasicType","MemberReference","TypeArgument"
                ,"TypeParameter","SuperMemberReference","VariableDeclarator","StatementExpression","CatchClauseParameter","VariableDeclaration",
               "Import",]
bop_connect = {
    "==": "boolean",
    "&&": "boolean",
    "||": "boolean",
    "!=": "boolean",
    ">": "boolean",
    ">=": "boolean",
    "<": "boolean",
    "<=": "boolean",
    #"instanceof": "boolean",
}
uop_connect = {
    #"!": "boolean",
}

type_dict = {}
func_dict = {}
member_dict = {}

def unify_if(if_node, cache_dict, upper_if_conds):
    my_conds = []
    my_stmts = []

    if_cond_seq = build_sequence_oast(if_node.condition, cache_dict)
    then_statement = if_node.then_statement
    if then_statement.__class__.__name__ == "IfStatement":
        upper_if_conds.append(if_cond_seq)
        inner_conds, inner_stmt = unify_if(then_statement, cache_dict, upper_if_conds)
        my_conds.extend(inner_conds)
        my_stmts.extend(inner_stmt)
    else:
        then_seq = build_sequence_oast(then_statement, cache_dict)
        if len(upper_if_conds) == 0:
            my_conds.append(if_cond_seq)
            my_stmts.append(then_seq)
        else:
            final_if_cond_seq = []
            final_if_conds = []
            final_if_conds.extend(upper_if_conds)
            final_if_conds.append(if_cond_seq)

            for i in range(len(final_if_conds) - 2, -1, -1):
                final_if_cond_seq.append("BinaryOperation_&&_boolean")
                if len(final_if_conds[i + 1]) + len(final_if_conds[i]) > 0:
                    final_if_cond_seq.append("[")
                    final_if_cond_seq.extend(final_if_conds[i + 1])
                    final_if_cond_seq.extend(final_if_conds[i])
                    final_if_cond_seq.append("]")

                final_if_conds[i] = copy.deepcopy(final_if_cond_seq)

            #print("IN", final_if_cond_seq, then_seq)
            my_conds.append(final_if_cond_seq)
            my_stmts.append(then_seq)

    if if_node.else_statement is not None:
        if len(upper_if_conds) > 0:
            final_else_cond_seq = []
            final_else_conds = []
            final_else_conds.extend(upper_if_conds)
            else_cond_seq = ["UnaryOperator_!_boolean"]
            if len(if_cond_seq) > 0:
                else_cond_seq.append("[")
                else_cond_seq.extend(if_cond_seq)
                else_cond_seq.append("]")
            final_else_conds.append(else_cond_seq)

            for i in range(len(final_else_conds) - 2, -1, -1):
                final_else_cond_seq.append("BinaryOperation_&&_boolean")
                if len(final_else_conds[i + 1]) + len(final_else_conds[i]) > 0:
                    final_else_cond_seq.append("[")
                    final_else_cond_seq.extend(final_else_conds[i + 1])
                    final_else_cond_seq.extend(final_else_conds[i])
                    final_else_cond_seq.append("]")

                final_else_conds[i] = copy.deepcopy(final_else_cond_seq)

            else_seq = build_sequence_oast(if_node.else_statement, cache_dict)
            my_conds.append(final_else_cond_seq)
            my_stmts.append(else_seq)
        else:
            else_cond_seq = ["UnaryOperator_!_boolean"]
            if len(if_cond_seq) > 0:
                else_cond_seq.append("[")
                else_cond_seq.extend(if_cond_seq)
                else_cond_seq.append("]")
            else_seq = build_sequence_oast(if_node.else_statement, cache_dict)
            #print("IN", else_cond_seq, else_seq)
            my_conds.append(else_cond_seq)
            my_stmts.append(else_seq)

    return my_conds, my_stmts

def try_unify(token, ast_node, cache_dict):
    sequence = []
    if token == "SwitchStatement":
        sequence.append("SelectionStatement")
        sequence.append("[")
        for i in range(len(ast_node.cases)):
            scase = ast_node.cases[i]
            switch_case = scase.case
            switch_case_seq = ["BinaryOperation_==_boolean"]
            switch_case_rvalue = build_sequence_oast(switch_case, cache_dict)
            if len(switch_case_rvalue) > 0:
                switch_case_seq.append("[")
                switch_case_seq.extend(switch_case_rvalue)
                switch_case_seq.append("]")
            switch_stmt = scase.statements
            switch_stmt_seq = build_sequence_oast(switch_stmt, cache_dict)
            sequence.extend(switch_case_seq)
            sequence.extend(switch_stmt_seq)
        sequence.append("]")
    # if token == "IfStatement":
    elif token == "IfStatement":
        token = "SelectionStatement"
        sequence.append(token)
        inner_seq = []
        conds, stmts = unify_if(ast_node, cache_dict, [])
        for i in range(len(conds)):
            inner_seq.extend(conds[i])
            inner_seq.extend(stmts[i])

        if len(inner_seq) > 0:
            sequence.append("[")
            sequence.extend(inner_seq)
            sequence.append("]")
        #print(sequence)
        count_l = 0
        count_r = 0
        for tok in sequence:
            if tok == "[":
                count_l += 1
            if tok == "]":
                count_r += 1
            if isinstance(tok, list):
                print(tok)
        if count_r != count_l:
            print(sequence)
            print(count_r, count_l)
        # print("COND", conds)
        # print("STMT", stmts)

        #print(ast_node)
    elif token == "ForStatement":
        control = ast_node.control
        if control.__class__.__name__ == "ForControl":
            loop_cond = control.condition
            loop_init = control.init
            loop_update = control.update

            if loop_init is not None:
                sequence.extend(build_sequence_oast(loop_init, cache_dict))

            token = "LoopStatement"
            sequence.append(token)
            sequence.append("[")
            if loop_cond is not None:
                sequence.extend(build_sequence_oast(loop_cond, cache_dict))

            loop_body = ast_node.body
            sequence.extend(build_sequence_oast(loop_body, cache_dict))

            if loop_update is not None:
                if sequence[-1] == "]":
                    sequence.pop()
                    sequence.extend(build_sequence_oast(loop_update, cache_dict))
                    sequence.append("]")
                else:
                    sequence.extend(build_sequence_oast(loop_update, cache_dict))

            sequence.append("]")
        else:
            enhanced_cond = build_sequence_oast(control, cache_dict)

            token = "LoopStatement"
            sequence.append(token)
            sequence.append("[")
            sequence.extend(enhanced_cond)
            loop_body = ast_node.body
            sequence.extend(build_sequence_oast(loop_body, cache_dict))
            sequence.append("]")

    elif token == "WhileStatement" or token == "DoStatement":
        token = "LoopStatement"
        sequence.append(token)

        loop_body = ast_node.body
        loop_cond = ast_node.condition
        loop_cond_seq = build_sequence_oast(loop_cond, cache_dict)
        loop_body_seq = build_sequence_oast(loop_body, cache_dict)
        if len(loop_cond_seq) + len(loop_body_seq) > 0:
            sequence.append("[")
            sequence.extend(loop_cond_seq)
            sequence.extend(loop_body_seq)
            sequence.append("]")

    return sequence

## build OAST sequence
def build_sequence_oast(node, cache_dict):
    sequence = []
    token, ast_node, token_unary = get_token_ast(node, cache_dict)
    if TO_UNIFY:
        #"SwitchStatement",
        if token in ["IfStatement", "ForStatement", "WhileStatement", "DoStatement", "SwitchStatement"]:
            sequence = try_unify(token, ast_node, cache_dict)
            return sequence

    children = get_child_ast(node, cache_dict)
    # Update Cache
    if token is not None and 'BinaryOperation' in token:
        cache_dict['in_BOP'] = True
    # Temp Symbol Table use for trace type in Java
    if token is not None and token == "FormalParameter":
        type_name = ast_node.type.name
        decl_name = ast_node.name
        if decl_name in cache_dict['symbol_table']:
            cache_dict['symbol_table'][decl_name].append(type_name)
        else:
            cache_dict['symbol_table'][decl_name] = [type_name]
    if token is not None and (token == "LocalVariableDeclaration" or token == "VariableDeclaration"):
        type_name = ast_node.type.name
        for decl in ast_node.declarators:
            decl_name = decl.name
            if decl_name in cache_dict['symbol_table']:
                cache_dict['symbol_table'][decl_name].append(type_name)
            else:
                cache_dict['symbol_table'][decl_name] = [type_name]

    if token_unary is not None:
        sequence.append(token_unary)

    is_append = False
    if token is not None and token is not "":
        if TO_CLEAN and token not in remove_node:
            is_append = True
            if token_unary is not None:
                sequence.append("[")
            sequence.append(token)

    # print(len(sequence), token)
    child_sequence = []
    for child in children:
        child_sequence.extend(build_sequence_oast(child, cache_dict))
    if len(child_sequence) >= 1:
        if is_append:
            sequence.append("[")
        for tok in child_sequence:
            sequence.append(tok)
        if is_append:
            sequence.append("]")

    if token_unary is not None and is_append:
        sequence.append("]")

    # ReUpdate Cache
    if token is not None and 'BinaryOperation' in token:
        cache_dict['in_BOP'] = False
    return sequence

def is_convertible_to_float_or_int(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def two_type_choose(lnode_type, rnode_type):
    basic_types = ["double", "float", "long", "Integer", "short", "byte", "char", "boolean", "NULL"]
    if lnode_type == rnode_type:
        type_name = lnode_type
    elif lnode_type != "Builtin" and rnode_type == "Builtin":
        type_name = lnode_type
    elif rnode_type != "Builtin" and lnode_type == "Builtin":
        type_name = rnode_type
    elif (lnode_type == "String" and rnode_type in basic_types) or (rnode_type == "String" and lnode_type in basic_types):
            type_name = "String"
    else:
        if (lnode_type == "double" and rnode_type in basic_types) or (rnode_type == "double" and lnode_type in basic_types):
            type_name = "double"
        elif (lnode_type == "float" and rnode_type in basic_types) or (rnode_type == "float" and lnode_type in basic_types):
            type_name = "float"
        elif (lnode_type == "long" and rnode_type in basic_types) or (rnode_type == "long" and lnode_type in basic_types):
            type_name = "long"
        elif (lnode_type == "Integer" and rnode_type in basic_types) or (rnode_type == "Integer" and lnode_type in basic_types):
            type_name = "Integer"
        elif (lnode_type == "short" and rnode_type in basic_types) or (rnode_type == "short" and lnode_type in basic_types):
            type_name = "short"
        else:
            #print(f"CONFLICT TYPE {lnode_type} | {rnode_type} ")
            type_name = lnode_type
    return type_name

def get_node_type(node, cache_dict):
    if node.__class__.__name__ == "MemberReference":
        member_name = node.member
        if member_name in cache_dict['symbol_table']:
            type_name = cache_dict['symbol_table'][member_name][-1]
        else:
            # print(cache_dict['symbol_table'])
            # print(f"UNKNOWN MEMBER {member_name}")
            type_name = "Builtin"
    elif node.__class__.__name__ == "BinaryOperation":
        lnode_type = get_node_type(node.operandl, cache_dict)
        rnode_type = get_node_type(node.operandr, cache_dict)
        type_name = two_type_choose(lnode_type, rnode_type)
    elif node.__class__.__name__ == "ClassCreator":
        type_name = node.type.name
    elif node.__class__.__name__ == "Literal":
        node_value = node.value
        if "\"" in node_value:
            type_name = "String"
        elif node_value == "null":
            type_name = "NULL"
        elif node_value == "true" or node_value == "false":
            type_name = "boolean"
        elif "\'" in node_value:
            type_name = "char"
        elif len(node_value) > 2 and node_value[0] == "0" and node_value[1] in ["x", "X"]:
            type_name = "int"
        elif node_value[-1] in ["L", "l"]:
            type_name = "long"
        elif node_value[-1] in ["f"]:
            type_name = "float"
        elif node_value[-1] in ["d"]:
            type_name = "double"
        elif is_convertible_to_float_or_int(node_value):
            if '.' in node_value:
                type_name = "float"
            else:
                type_name = "int"
        else:
            #print(f"NONE Literal {node_value}")
            type_name = "Builtin"
    elif node.__class__.__name__ == "MethodInvocation":
        func_member = node.member
        if func_member in cache_dict["common_func"]:
            type_name = func_member
        else:
            type_name = "Builtin"
    elif node.__class__.__name__ == "Cast":
        type_name = node.type.name
    elif node.__class__.__name__ == "This":
        if len(node.selectors) > 0:
            type_name = get_node_type(node.selectors[0], cache_dict)
        else:
            type_name = "Builtin"
    elif node.__class__.__name__ == "ReferenceType":
        type_name = node.name
    elif node.__class__.__name__ == "BasicType":
        type_name = node.name
    elif node.__class__.__name__ == "FormalParameter":
        type_name = node.type.name
    elif node.__class__.__name__ == "CatchClause":
        type_name = node.parameter.types[0]
        if "." in type_name:
            type_name = type_name.split(".")[-1]
    elif node.__class__.__name__ == "TernaryExpression":
        lnode_type = get_node_type(node.if_true, cache_dict)
        rnode_type = get_node_type(node.if_false, cache_dict)
        type_name = two_type_choose(lnode_type, rnode_type)
    elif node.__class__.__name__ == "Assignment":
        type_name = get_node_type(node.value, cache_dict)
    elif node.__class__.__name__ == "ClassReference":
        type_name = node.type.name
    elif node.__class__.__name__ == "ArrayCreator":
        type_name = "ArrayList"
    elif node.__class__.__name__ == "NoneType":
        type_name = "Builtin"
    else:
        #print(f"NONE CLASS {node.__class__.__name__}")
        type_name = "Builtin"

    if type_name == "int":
        type_name = "Integer"
    if type_name == "Long":
        type_name = "long"
    if type_name == "Double":
        type_name = "double"

    if type_name not in type_dict:
        type_dict[type_name] = 0
    type_dict[type_name] += 1

    if type_name not in cache_dict["common_type"]:
        type_name = "Builtin"

    return type_name

def get_token_ast(node, cache_dict):
    token = ''
    ast_node = None
    token_unary = None
    if isinstance(node, str):
        token = None
    elif isinstance(node, set):
        # token = 'Modifier'
        token = None
    elif isinstance(node, Node):
        ast_node = node
        try:
            pre_op = node.prefix_operators
            post_op = node.postfix_operators
            if len(pre_op) > 0 or len(post_op) > 0:
                if TO_ENHANCE:
                    if len(pre_op) > 0:
                        op = pre_op[0]
                    elif len(post_op) > 0:
                        op = post_op[0]
                    if op in uop_connect:
                        type_name = uop_connect[op]
                    else:
                        type_name = get_node_type(node, cache_dict)
                    token_unary = f"UnaryOperator_{op}_{type_name}"
                else:
                    token_unary = "UnaryOperator"
        except Exception as e:
            pass
        token = node.__class__.__name__
        if "BinaryOperation" in token and TO_ENHANCE:
            op = node.operator
            if op in bop_connect:
                type_name = bop_connect[op]
            else:
                type_name = get_node_type(node, cache_dict)
            token = f"{token}_{op}_{type_name}"
        if "TernaryExpression" in token and TO_ENHANCE:
            type_name = get_node_type(node, cache_dict)
            token = f"{token}_{type_name}"
        if "Assignment" in token and TO_ENHANCE:
            type_name = get_node_type(node, cache_dict)
            token = f"BinaryOperation_=_{type_name}"
        if token == "ClassCreator" and TO_ENHANCE:
            type_name = get_node_type(node, cache_dict)
            token = f"ClassCreator_{type_name}"
        # if token == "MemberReference" and TO_ENHANCE:
        #     node_member = ast_node.member
        #     if node_member not in member_dict:
        #         member_dict[node_member] = 0
        #     member_dict[node_member] += 1
        #     type_name = get_node_type(node, cache_dict)
        #     if type_name in common_type:
        #         token = f"{token}_{type_name}"
        #     else:
        #         token = f"{token}"
        if token == "MethodDeclaration" and TO_ENHANCE:
            if ast_node.return_type:
                type_name = ast_node.return_type.name
            else:
                type_name = "void"
            token = f"MethodDeclaration_{type_name}"
        if token == "FormalParameter" and TO_ENHANCE:
            type_name = ast_node.type.name
            token = f"FormalParameter_{type_name}"
        if token == "CatchClause" and TO_ENHANCE:
            type_name = get_node_type(node, cache_dict)
            token = f"CatchClause_{type_name}"
        if token == "MethodInvocation" and TO_ENHANCE:
            func_member = ast_node.member
            if func_member not in func_dict:
                func_dict[func_member] = 0
            func_dict[func_member] += 1
            if func_member in cache_dict["common_func"]:
                token = f"{token}_{func_member}"
            else:
                token = f"{token}"
        # if "MemberReference" in token and TO_ENHANCE:
        #     print(node)
    return token, ast_node, token_unary

def get_child_ast(root, cache_dict):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))

def save_type_dict():
    sorted_dict = dict(sorted(type_dict.items(), key=lambda item: item[1], reverse=True))
    amount_type = []
    total = 0
    inlist = 0
    with open("OAST_type.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            total += value
            if value > 8:
                inlist += value
                amount_type.append(f"\"{key}\"")
            file.write(f"{key}: {value}\n")
    type_list_str = f'[{",".join(amount_type)}]'
    print(f"type total:inlist = {total} : {inlist} : {total - inlist}")
    with open("OAST_type_list.txt", "w", encoding="utf-8") as file:
        file.write(type_list_str)

    amount_type = []
    total = 0
    inlist = 0
    sorted_dict = dict(sorted(func_dict.items(), key=lambda item: item[1], reverse=True))
    with open("OAST_func.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            total += value
            if value > 8:
                inlist += value
                amount_type.append(f"\"{key}\"")
            file.write(f"{key}: {value}\n")
    type_list_str = f'[{",".join(amount_type)}]'
    print(f"func total:inlist = {total} : {inlist} : {total - inlist}")
    with open("OAST_func_list.txt", "w", encoding="utf-8") as file:
        file.write(type_list_str)

    amount_type = []
    total = 0
    inlist = 0
    sorted_dict = dict(sorted(member_dict.items(), key=lambda item: item[1], reverse=True))
    with open("OAST_member.txt", "w", encoding="utf-8") as file:
        for key, value in sorted_dict.items():
            total += value
            if value >= 8:
                inlist += value
                amount_type.append(f"\"{key}\"")
            file.write(f"{key}: {value}\n")
    type_list_str = f'[{",".join(amount_type)}]'
    print(f"member total:inlist = {total} : {inlist} : {total - inlist}")
    with open("OAST_member_list.txt", "w", encoding="utf-8") as file:
        file.write(type_list_str)

