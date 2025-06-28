import os
print('Start ReVeal experiments in Table 2')
print('Running Table 2 - ReVeal - Reproduction - Baseline')
p=os.popen('python devign_non_vulgen_reveal_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Reproduction - Synthetic')
p=os.popen('python devign_vulgen_syn_reveal_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Reproduction - Generated')
p=os.popen('python devign_vulgen_reveal_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Reproduction - Ground Truth')
p=os.popen('python devign_vulgen_gth_reveal_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Reproduction - Wild')
p=os.popen('python devign_wild_reveal_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)



print('Running Table 2 - ReVeal - Replication - Baseline')
p=os.popen('python devign_non_vulgen_xen_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Replication - Synthetic')
p=os.popen('python devign_vulgen_syn_xen_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Replication - Generated')
p=os.popen('python devign_vulgen_xen_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Replication - Ground Truth')
p=os.popen('python devign_vulgen_gth_xen_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)

print('Running Table 2 - ReVeal - Replication - Wild')
p=os.popen('python devign_wild_xen_api_test.py 2> /dev/null')
output=p.read()
print('Accuracy:         Precision:         Recall:           F1:')
print(output)
