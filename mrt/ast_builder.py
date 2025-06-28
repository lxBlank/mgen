from javalang.ast import Node

unary_op = ['!','~','++','--','-']
def get_token_ast(node, cache_dict):
    token = ''
    if isinstance(node, str):
        if node in unary_op and not cache_dict['in_BOP']:
            token = "UnaryOperator"
        else:
            token = None
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
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
    #Update Cache
    if token == 'BinaryOperation':
        cache_dict['in_BOP'] = True

    if token is not None and token is not "":
        sequence.append(token)
    #print(len(sequence), token)
    child_sequence = []
    for child in children:
        child_sequence.extend(get_sequence_ast(child, cache_dict))
    if len(child_sequence) >= 1:
        sequence.append("[")
        for tok in child_sequence:
            sequence.append(tok)
        sequence.append("]")

    # ReUpdate Cache
    if token == 'BinaryOperation':
        cache_dict['in_BOP'] = False
    return sequence

