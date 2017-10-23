import os 


def convert_encoding():
	in_file = '/projects/kawa8312/word2vec/word2vec/data/merge_genia_300k'
	out_file = 'merge_genia_300k_utf8'
	fi = open(in_file,'r')
	fo = open(out_file, 'wb')
	
	count = 0 
	for line in fi:
		count +=1 
		print "processing line :: ",count
		line_utf = line.encode("utf-8")
		fo.write(line)
	
	fi.close()
	fo.close()
	
	
