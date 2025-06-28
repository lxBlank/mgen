from javalang.ast import Node

block_stmts = ["ForStatement", "ForEachStatement", "IfStatement", "WhileStatement", "TryStatement", "SynchronizedStatement", "SwitchStatement", "DoStatement"]
ignore_nodes = ["PackageDeclaration", "Import", "Literal"]
def get_token_ast(node, cache_dict):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #if token is not None:
    return token

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

def get_sequence_ast(node, cache_dict):
    sequence = []
    token, children = get_token_ast(node, cache_dict), get_child_ast(node, cache_dict)
    if token is not None and token is not "" and token not in ignore_nodes and " " not in token:
        sequence.append(token)
        child_sequence = []
        for child in children:
            child_sequence.extend(get_sequence_ast(child, cache_dict))
        if len(child_sequence) >= 1:
            sequence.append("[")
            for tok in child_sequence:
                sequence.append(tok)
            sequence.append("]")
    return sequence

def build_sequence_sast(node, cache_dict):
    sequence = []
    sequence.append("Root")
    sequence.append("[")
    sequence.extend(get_sequence_sast(node, cache_dict))
    sequence.append("]")
    return sequence

def get_sequence_sast(node, cache_dict):
    token, children = get_token_ast(node, cache_dict), get_child_ast(node, cache_dict)
    sequence = []
    if token is not None and token is not "" and token not in ignore_nodes:
        body = []
        block = []
        #split body and block
        if token == "CompilationUnit":
            block_children = get_child_ast(node, cache_dict)
            for bchild in block_children:
                body.append(bchild)
        else:
            body_statements = []
            if hasattr(node, 'body') and node.body:
                body_statements = list(node.body)
            for child in children:
                if child in body_statements and get_token_ast(child, cache_dict) in block_stmts:
                    body.append(child)
                else:
                    block.append(child)
        #block
        inner_sequence = []
        block_sequence = []
        for child in block:
            block_sequence.extend(get_sequence_ast(child, cache_dict))
        if len(block_sequence) >= 1:
            for tok in block_sequence:
                inner_sequence.append(tok)
        #body
        for child in body:
            bchild_seqs = get_sequence_sast(child, cache_dict)
            if len(bchild_seqs) > 0:
                inner_sequence.extend(bchild_seqs)
        sequence.append(token)
        if len(inner_sequence) > 0:
            sequence.append("[")
            sequence.extend(inner_sequence)
            sequence.append("]")
    return sequence

