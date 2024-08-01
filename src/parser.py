import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='MSDS')
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='ACGSL',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--epochs',
                    type=int,
					required=False,
					default=5,   # ACGSL MSDS 1 best
					help='set the trained epochs: SWaT_va: 7 other 5')
parser.add_argument('--top_k',
                   type=int,
				   required=False,
				   default=5,
				   help='select the number of edge for each node')		
parser.add_argument('--noise_std',
                    type=float,
					required=False,
					default=0.0,
					help='add gaussion noise to data, noise~(0,std)')
parser.add_argument('--GNN',
                    type=str,
					required=False,
					default='GIN',
					help='Select the GNN model to aggregate the node features' )
parser.add_argument('--device',
                    type=str,
					default='cuda',
					help='cuda / cpu')
parser.add_argument('--ls_0',
                    type=float,
					default=1.0,
					help='super-parameter for construction loss ')
parser.add_argument('--ls_1',
                    type=float,
					default=0.1,
					help='super-parameter for construction loss ')
parser.add_argument('--ls_2',
                    type=float,
					default=0.001,
					help='super-parameter for construction loss ')

args = parser.parse_args()