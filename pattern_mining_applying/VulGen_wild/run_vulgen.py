import os
import pickle
import pandas as pd

print('Running VulGen for wild samples (Table 2 last column)')
os.chdir('/root/VulGen/T5/M1_VulRepair_PL-NL')
print('Start vulnerability injection localization.')
print('The expected running time is 15 min')
p=os.popen('python vulrepair_wild.py --output_dir=./saved_models --model_name=model.bin --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --do_test --encoder_block_size 512 --decoder_block_size 512 --num_beams=1 --eval_batch_size 1 2> wild_beam1.log')
print('You can run "tail -f /root/VulGen/T5/M1_VulRepair_PL-NL/wild_beam1.log" in another terminal for the progress.')
output=p.read()
os.chdir('/root/VulGen/pattern_mining_applying/VulGen_wild')
os.system('cp /root/VulGen/T5/data/raw_predictions/CodeT5/VulRepair_raw_preds_wild_beam1.csv ./')
df=pd.read_csv('VulRepair_raw_preds_wild_beam1.csv')
obj=df.to_dict()
f=open('VulRepair_raw_preds_wild_beam1.pkl','wb')
pickle.dump(obj,f)
f.close()
print('Finished localization. You can check "VulRepair_raw_preds_final_beam1.csv" for the localization result.')

print('\nStart applying injection patterns.')
print('The expected running time is 5 min')
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
print('The generated samples can be found in pattern_mining_applying/VulGen_wild/generated')
print('The log file can be found in pattern_mining_applying/VulGen_wild/res.txt')
print('The generated samples are used for data augmentation in Table 2 last column')



