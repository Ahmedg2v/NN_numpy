import numpy as np
import random

def Batch_gen(t_data,t_labels,batch_size):
    data=[]
    labels=[]
    
    for m in range(batch_size):
        idx = random.choice(range(len(t_data)))
        data.append(t_data[idx])
        labels.append(t_labels[idx])

    data=np.array(data)
    labels=np.array(labels)
    
    return data,labels
