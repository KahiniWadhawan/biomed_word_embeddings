import os
import sys
import argparse
from bs4 import BeautifulSoup as Soup

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='Extarct abstract text from Pubmed XMLs')
    parser.add_argument('--part_id', required=True, type=int, default=0,
                        help='Part Id in dataset {0-8}')
    args = vars(parser.parse_args(args))
    return args

def save_abstracts(part_id):
    data_dir = '../data/baseline_parts/part' + str(part_id)
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)
    text_data_dir = '../text/baseline_parts/part' + str(part_id)
    text_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), text_data_dir)
    if not os.path.exists(text_data_dir):
        os.makedirs(text_data_dir)

    print("Working on dataset part_id - " + str(part_id))
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files if f.endswith('.xml')]
    print("Total number of files are - " + str(len(files)))
    # print(files)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        print("Working on file #" + str(index) + " - " + filename)
        file_handle = open(file, encoding='utf8')
        file_data = file_handle.read()
        text_file_name =  filename + '.text'
        text_file_name = os.path.join(text_data_dir, text_file_name)
        output_file_handle = open(text_file_name, 'w', encoding='utf8')
        soup = Soup(file_data, "lxml-xml")
        for message in soup.findAll('AbstractText'):
            # print(message.text)
            output_file_handle.write(message.text + "\n")
            output_file_handle.flush()
        output_file_handle.close()
        file_handle.close()
        sys.stdout.flush()

def main(args):
    args = parse_args(args)
    save_abstracts(args['part_id'])

if __name__ == '__main__':
    main(sys.argv[1:])