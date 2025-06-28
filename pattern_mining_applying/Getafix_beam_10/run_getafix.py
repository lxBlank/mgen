import os
import pickle
import pandas as pd
print('\n\n\nRunning Getafix (Table 1 Row 4 Column 3) Success Rate')

print('Start applying injection patterns.')
print('The expected running time is 15 min')
print('Notice that to reduce the time and memory cost, we pre-compute the specialization score in formula 5 so that you do not need to spend the time shown in RQ6.')
try:
    os.mkdir('generated')
except:
    pass
p=os.popen('pypy3 testApplying.py')
output=p.read()
f=open('res.txt','w')
f.write(output)
f.close()
#succ=output.count('success')
#p=os.popen('ls -l generated/*_gen.c|wc -l')
#total=int(p.read())
print('The generated samples can be found in pattern_mining_applying/Getafix_beam_10/generated')
print('The log file can be found in pattern_mining_applying/Getafix_beam_10/res.txt')
print('The results need manual check. You can find our manual check in pattern_mining_applying/Getafix_beam_10/manual_inspection')



