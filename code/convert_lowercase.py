import argparse

def convert_case(ip_file, op_file):
	with open(op_file, 'w') as opf:
		with open(ip_file,'r') as ipf: 
			for line in ipf:
 				opf.write(line.lower())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ipf', type=str, required=True, help='Input file path')
  parser.add_argument('--opf', type=str, required=True, help='Output file path')
  args = parser.parse_args()
  convert_case(args.ipf, args.opf)
