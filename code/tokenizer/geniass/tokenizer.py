import os
from os import path
from subprocess import call
import argparse
import sys

def run_tokenizer(args):
	directory_path = args.ip_dir 
	output_path = args.op_dir
	for x in os.listdir(directory_path):
		filename = os.path.join(directory_path, x) 
		if path.isfile(filename):
			print('found file :: ', filename)
                	new_x = filename.split('/')[-1]
			output_filename = os.path.join(output_path, new_x)
			print('output file :: ', output_filename)	
			command = "./tokenizer " + filename + " " + output_filename
			os.system(command)
			sys.stdout.flush()
			
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ip_dir', type=str, required=True, help='Input directory path')
  parser.add_argument('--op_dir', type=str, required=True, help='Output directory path')
  args = parser.parse_args()
  run_tokenizer(args)
