import os
from os import path
from subprocess import call
import shutil

directory_path = "/lustre/janus_scratch/kawa8312/pubmed_complete_flat"
output_path = "/lustre/janus_scratch/kawa8312/pubmed_complete_flat_tokenized"

count = 0
new_filename = ""
for x in os.listdir(directory_path):
	filename = directory_path + os.sep + x
	if path.isfile(filename):
		if filename.endswith(".txt"):
			if '(' in filename or ')' in filename:
				new_filename = filename.replace('(', '-')
				new_filename = new_filename.replace(')', '-')
				shutil.copyfile(filename, new_filename)
			else:
				new_filename = filename
			new_x = x.replace('(', '-')
			new_x = new_x.replace(')', '-')
			output_filename = os.path.join(output_path, new_x)
			
			# Check if the output file is already present, then continue
			# Needed because this is taking a long time and multiple jobs
			if path.isfile(output_filename):
				print "Already exists - " + output_filename
				continue

			command = "./tokenizer " + new_filename + " " + output_filename
			#print filename + "\n"
			#print output_filename + "\n"
			#print command + "\n"
			os.system(command)
			if count % 40000 == 0:
				print count
			count = count + 1
			
		
