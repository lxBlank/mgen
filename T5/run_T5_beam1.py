import os
import pickle
import pandas as pd
print('\n\n\nRunning T5 (Table 1 Row 3 Column 2)')
os.chdir('/root/VulGen/T5/M1_VulRepair_PL-NL')
print('Start vulnerability injection using T5.')
print('The expected running time is 45 min')

p=os.popen('python vulrepair_T5_beam1.py --output_dir=./saved_models --model_name=model.bin --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --do_test --encoder_block_size 512 --decoder_block_size 512 --num_beams=1 --eval_batch_size 1 2> T5_beam1.log')
print('You can run "tail -f /root/VulGen/T5/M1_VulRepair_PL-NL/T5_beam1.log" in another terminal for the progress.')
output=p.read()

f=open('T5_beam1.log')
lines=f.readlines()
print(lines[-1].split('-')[-1][:-1])
print('The accuracy may have a small differnce because of the randomness of the model.')

