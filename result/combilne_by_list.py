import os
import pandas as pd
import tqdm
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input Date')
    parser.add_argument('--option', type=int, help='0: all',default=0)
    parser.add_argument('--file_name', type=str, help='file_name',default='temp')
    args = parser.parse_args()
    option = args.option
    file_name = args.file_name

    if option==0:
        file_names = os.listdir()
        file_names = [file for file in file_names if '.csv' in file]

        total_list = []
        for file in tqdm.tqdm(file_names):
            try: 
                temp = pd.read_csv(file,encoding='cp949',low_memory=False)
                total_list.append(temp)
            except pd.errors.EmptyDataError:
                print(file)
        df_accum = pd.concat(total_list)
        df_accum.to_csv(f'{file_name}.csv',encoding='cp949',index=False)