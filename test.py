from argparse import ArgumentParser
parser = ArgumentParser()

# arguments
parser.add_argument(
        '--experiment_name', 
        default=1,   
        type=int,          
        help='Define the project name.',
    )
args = parser.parse_args()
args.aa = 2222
print(args.aa)