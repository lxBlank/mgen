import os
import pickle
import pandas as pd
print('\n\n\nRunning Graph2Edit (Table 1 Row 5 Column 3)')
print('Start vulnerability injection using Graph2Edit.')
print('The expected running time is 4 hours')
print('You can run "tail -f /root/VulGen/g2e_vulgen/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220810-152940/test_beam10.*" in another terminal for the progress.')
os.system('bash scripts/githubedits/test_vulgen_beam10.sh')
print('Vulnerability injection done')
print('The generated samples can be found in g2e_vulgen/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220810-152940/model.bin.decode_beam10_jsonl.json')
print('The results need manual check. You can find our manual check in g2e_vulgen/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220810-152940/success')


