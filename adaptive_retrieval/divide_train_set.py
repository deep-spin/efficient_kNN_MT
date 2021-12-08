import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str)
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()

file_de = open(args.data+'.de','r')
file_en = open(args.data+'.en','r')

file_train_de = open(args.out_dir+'train.de','w')
file_train_en = open(args.out_dir+'train.en','w')

file_train_datastore_de = open(args.out_dir+'datastore_set/train.de','w')
file_train_datastore_en = open(args.out_dir+'datastore_set/train.en','w')

lines_de = file_de.readlines()
lines_en = file_en.readlines()

data_size = len(lines_de)
train_size = int(.5*data_size)
train_datastore_size = data_size-train_size

train_lines = np.random.choice(data_size,size=train_size,replace=False).tolist()

for i in range(data_size):
	if i in train_lines:
		file_train_de.write(lines_de[i])
		file_train_en.write(lines_en[i])
	else:
		file_train_datastore_de.write(lines_de[i])
		file_train_datastore_en.write(lines_en[i])


