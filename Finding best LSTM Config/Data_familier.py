import numpy as np
import pandas as pd

noised  = pd.read_csv(r"C:\Users\Mahsa\Desktop\project Msc\code denoise\EcgDenoising_recurrent LSTM\noised.csv")
print(noised.shape)  #(499, 1)

original = pd.read_csv(r"C:\Users\Mahsa\Desktop\project Msc\code denoise\EcgDenoising_recurrent LSTM\original.csv")
#print(original.shape) #(499, 1)

###Real dataset
DataSet_npy1 = np.load(r"C:\Users\Mahsa\Desktop\project Msc\code denoise\EcgDenoising_recurrent LSTM\real_dataset\dataset.npy")
#print(DataSet_npy1.shape) # (549, 30000)

DataSet_Normalized = np.load(r"C:\Users\Mahsa\Desktop\project Msc\code denoise\EcgDenoising_recurrent LSTM\real_dataset\dataset_normalized.npy")
#print(DataSet_Normalized) # (2, 1000, 300, 1)

