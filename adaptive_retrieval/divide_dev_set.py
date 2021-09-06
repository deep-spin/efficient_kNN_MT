import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str)
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()

file_de = open(args.data+'.de','r')
file_en = open(args.data+'.en','r')

file_train_de = open(args.out_dir+'_train.de','r')
file_train_en = open(args.out_dir+'_train.en','r')

file_dev_de = open(args.out_dir+'_dev.de','r')
file_dev_en = open(args.out_dir+'_dev.en','r')

lines_de = file_de.readlines()
lines_en = file_en.readlines()

data_size = len(lines)
train_size = int(.9*len(lines))
dev_size = data_size-train_size

train_lines = np.random.randint(data_size,train_size)[0]

for i in range(data_size):
	if i in train_lines:
		file_train_de.write(lines_de[i])
		file_train_en.write(lines_en[i])
	else:
		file_dev_de.write(lines_de[i])
		file_dev_en.write(lines_en[i])

