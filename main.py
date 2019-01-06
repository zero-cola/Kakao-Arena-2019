import json
import re
import os
import fire

MODEL_BASE_DIR = 'model'

def ps():

    model_path_list = [os.path.join(MODEL_BASE_DIR, name) for name in os.listdir(MODEL_BASE_DIR)]
    
    for model_path in model_path_list:
        print('\n>> Model Path: ',model_path)
        status_path_list = [os.path.join(model_path, name) for name in os.listdir(model_path) if 'status' in name]
        status_path_list.sort(key=lambda path: int(re.findall('\d+', path)[-1]))

        columns = "[{epoch:^2}] {time:^19}  {batch_size:^4}  {lr:^9}  {t_loss:^6}  {v_loss:^6}  {t_score:^6}  {v_score:^6}".format(
                    epoch='ep', time='time', batch_size='bs', lr='lr', t_loss='t_loss', v_loss='v_loss', t_score='t_score', v_score='v_score')
        print(columns)
        print('-'*len(columns))
        for status_path in status_path_list:
            status = json.load(open(status_path, 'r'))
            status['epoch'] = re.findall('\d+', status_path)[-1]
            
            fmt = "[{epoch:>2}] {time}  {batch_size:4}  {lr:.03e}  {t_loss:^6.4f}  {v_loss:^6.4f}  {t_score:^6.5f}  {v_score:^6.5f}"
            print(fmt.format(**status))

if __name__ == "__main__":
    fire.Fire({'ps':ps})

