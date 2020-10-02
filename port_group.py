import argparse
from modules.MetadataProcessor import MetadataProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Set value for data_dir")

def main(data_dir):
    
    processor = MetadataProcessor(data_dir, 10)
    processor.port_group()
    return

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))

