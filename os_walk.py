import os
import random
import shutil
from shutil import copyfile
import io
  

#def copyFiles():
 #       src_path = "/projects/kawa8312/toy_data"
#	src_file = getRandomFile(src_path)
 #       src = os.path.join(src_path,src_file) 
#	dest = "/projects/kawa8312/toy_data_copy" + src_file
 #       copyfile(src, dest)

def convert_encoding(src_path, dest_root):
	all_files = gen_dirs_tree(src_path)
        file_count = 0 
	for file in all_files:
		file_count += 1
		print "processing file :: ", file_count
		#fi = open(file,'r')
		file_split = file.split('/')
		out_filename = file_split[len(file_split)-1]
		out_file = os.path.join(dest_root,out_filename)
        	fo = open(out_file, 'wb')
		with io.open(file, "r", encoding="utf-8") as fi:
        		count = 0
        		for line in fi:
                		count +=1
				if count % 100 == 0: 
                			print "processing line :: ",count
            		    	line_utf = line.encode("ISO-8859-1")
                		fo.write(line)

        	fi.close()
        	fo.close()





def copy_rand_file(src_path,dest_root,num_files):
	#dest_root = "/projects/kawa8312/toy_data_copy"
        #Returns a random filename, chosen among the files of the given path.
        all_files = gen_dirs_tree(src_path)
        #files_exists = 0
        #extracting all rand_files
        for i in xrange(num_files+1):
                if i%1000 == 0:
                        print("processing ",i)
                index = random.randrange(0, len(all_files))
                rand_file = all_files[index]
                file_split = rand_file.split('/')
                rand_filename = file_split[len(file_split)-1]
                src = rand_file
                dest = os.path.join(dest_root,rand_filename)
                #copying random file to copy folder
                #       files_exists += 1
                #       i = i - 1
                #else:
                copyfile(src,dest)

def make_flat_dir(src_path,dest_root):
	#dest_root = "/projects/kawa8312/toy_data_copy"
	#Returns a random filename, chosen among the files of the given path.
 	all_files = gen_dirs_tree(src_path)
	#files_exists = 0 
	#extracting all rand_files 
	for i,file in enumerate(all_files):
		if i%500 == 0:
			print "processing :: ",i
		file_split = file.split('/')
		filename = file_split[len(file_split)-1]
		src = file 
		dest = os.path.join(dest_root,filename)
		copyfile(src,dest)


def gen_dirs_tree(path):
	all_files = []
	for root, dirs, files in os.walk(path, topdown=False):
     		#print "rand :: ", random.choice(files)
     		#print len(files)
     		for name in files:
			if name.endswith(".txt"):
        			f = os.path.join(root, name)
				all_files.append(f)
				#print "filename:: ",f
    	#for name in dirs:
     	#print(os.path.join(root, name))

	#print len(all_files)
	return all_files
	#print random.choice(files)

#src_path = "/projects/kawa8312/KahiniCode-ver2/data/tokenizedFiles/TextWinTokenFiles"
#dest_root = "/projects/kawa8312/KahiniCode-ver2/data/tokenizedFiles/TextWinTokenFiles_copy"
src_path = "pubmed_complete_flat_tokenized"
dest_root = "pubmed_complete_flat_tokenized_ISO_8859_1"
#copy_rand_file(src_path,dest_root,500000)
#make_flat_dir(src_path,dest_root)
convert_encoding(src_path,dest_root)
