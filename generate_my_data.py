import json
import pandas as pd
import numpy as np

############Generate a dummy dataset to mimic the true dataset##############################################

prefix = """Act as an experienced pathologist. 
Answer the question using the pathology report below. Base the answer on the report only. Do not add any additional information.
"""

question_1 = """
### Question: 
What is the tissue source? Answer 1 if left breast, 2 if right breast, 3 if left lymph node, 4 if right lymph node, 5 if other. If multiple sources are present, separate with comma. Give me a number only. Do not explain.            
### Answer:
"""
question_2 = """
### Question:
Is cancer found in the tissue examined? Answer 1 if yes, 0 if no, 2 if it's of high risk for cancer. Keep in mind that carcinoma is considered a cancer, but lobular carcinoma in situ (LCIS) is not. Give me a single number only. Do not explain.            
### Answer:
"""

lateralities = ["left breast", "right breast"]
types = ["ductal carcinoma in situ (DCIS)", "invasive ducal carcinoma (IDC)", "invasive lobular carcinoma (ILC)", "other invasive", "adenocarcinoma", "other"]
combs = {}  # a dictionary of all 12 prompts. key is index 1-12
k=0
for l in lateralities:
  for t in types:
    k+=1
    comb = f"""
    ### Question:
    If {l} is being examined, is there cancer of type {t} found in {l}? Answer 1 if yes, 0 if no.
    ### Answer: 
    """
    combs[k] = comb

# randomly generate 100 dummy reports, select 40 to have cancer (i.e. ask all 3 questions)
from random import choice, randint
from string import ascii_lowercase as characters
from string import digits

def generate_dataset(size, save_path, split):
  df = pd.DataFrame(columns=["Accession Number", "Report", "Prompt", "Response", "Question", "Input"])
  for i in range(size):
    df_current = pd.DataFrame(columns=["Accession Number", "Report", "Prompt", "Response", "Question", "Input"])
    count = randint(500, 1000)
    report = ''.join(choice(characters) for _ in range(count))
    report = f"""
    ### Pathology Report: 
    {report}"""
    id = ''.join(choice(digits) for _ in range(8))

    label_1 = np.random.choice(np.arange(1, 6), p=np.ones(5) / 5)
    label_2 = np.random.choice(np.arange(3), p=[0.3, 0.6, 0.1])
    if split=="test":
      df_current.loc[0, :] = [id, report, question_1, label_1, 1, prefix + report + question_1]
      df_current.loc[1, :] = [id, report, question_2, label_2, 2, prefix + report + question_2]
    else:
      df_current.loc[0, :] = [id, report, question_1, label_1, 1, prefix + report + question_1 + str(label_1)]
      df_current.loc[1, :] = [id, report, question_2, label_2, 2, prefix + report + question_2 + str(label_2)]
    
    if label_2 == 1:
      type_idx = np.random.choice(np.arange(k), p=np.ones(k) / k)
      for j in range(k):
        if k == type_idx:
          label_3 = 1
        else:
          label_3 = 0
        if split=="test":
          df.loc[df.shape[0], :] = [id, report, combs[j+1], label_3, 3, prefix + report + combs[j+1]]
        else:
          df.loc[df.shape[0], :] = [id, report, combs[j+1], label_3, 3, prefix + report + combs[j+1] + str(label_3)]
    df = pd.concat([df, df_current])
      
  print(f'\nTotal dimension of formatted dataset: {df.shape}\n')
  df.to_csv(save_path)
  return df

df_tr = generate_dataset(100,"./my_data/dummy-tr.csv", "train")
df_val = generate_dataset(30,"./my_data/dummy-val.csv", "val")
df_te = generate_dataset(30,"./my_data/dummy-te.csv", "test")

