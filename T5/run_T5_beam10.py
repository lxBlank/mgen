import os
import pickle
import pandas as pd
print('\n\n\nRunning T5 (Table 1 Row 3 Column 3)')
os.chdir('/root/VulGen/T5/M1_VulRepair_PL-NL')
print('Start vulnerability injection using T5.')
print('The expected running time is 2 hours')
p=os.popen('python vulrepair_T5_beam10.py --output_dir=./saved_models --model_name=model.bin --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --do_test --encoder_block_size 512 --decoder_block_size 512 --num_beams=10 --eval_batch_size 1 2> T5_beam10.log')
print('You can run "tail -f /root/VulGen/T5/M1_VulRepair_PL-NL/T5_beam10.log" in another terminal for the progress.')
output=p.read()
print('The generated samples can be found in T5/data/raw_predictions/CodeT5/VulRepair_raw_preds_translate_final_beam10.csv')
print('The results need manual check. You can find our manual check in T5/manual_inspection_beam_10')
