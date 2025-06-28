import os
import pickle
import pandas as pd
print('\n\n\nRunning Graph2Edit (Table 1 Row 5 Column 2)')
print('Start vulnerability injection using Graph2Edit.')
print('The expected running time is 45 min')
print('You can run "tail -f /root/VulGen/g2e_vulgen/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220810-152940/test_beam1.*" in another terminal for the progress.')
os.system('bash scripts/githubedits/test_vulgen_beam1.sh')
f=open('./exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220810-152940/test_beam1.log')
lines=f.readlines()
for line in lines:
    if 'acc@1=' in line:
        print(line)

