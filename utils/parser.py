from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--num',
                        dest='num', 
                        type=int, 
                        default=100)
    parser.add_argument('--logging', 
                        dest='logging',
                        action='store_true', 
                        default=False)
    parser.add_argument('--nologging', 
                        dest='logging',
                        action='store_false', 
                        default=False)

    return parser.parse_args()