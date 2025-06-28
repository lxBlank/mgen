import os
import pickle
import pandas as pd
print('\n\n\nRunning Getafix (Table 1 Row 4 Column 2)')

print('Notice that since Getafix randomly select a statement to inject the vulnerability, the accuracy may be different from the one shown in the paper. See Section V.E for details.')
print('Start applying injection patterns.')
print('The expected running time is 5 min')
print('Notice that to reduce the time and memory cost, we pre-compute the specialization score in formula 5 so that you do not need to spend the time shown in RQ6.')

try:
    os.system('rm -r generated')
except:
    pass

os.mkdir('generated')

p=os.popen('pypy3 testApplying.py')
output=p.read()
f=open('res.txt','w')
f.write(output)
f.close()

succ=output.count('success')
p=os.popen('ls -l generated/*_gen.c|wc -l')
total=int(p.read())+2
print('VulGen Precision (Exactly-Matched) (Table 1 Row 4 Column 2):', (1.0*succ+2)/total)
print('The log file can be found in pattern_mining_applying/Getafix_beam_1/res.txt')
