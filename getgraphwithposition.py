from data.dataset import PygPCQM4MDatasetWithPosition, smiles2graphWith2Dposition

def main():

    dataset = PygPCQM4MDatasetWithPosition(root = '../dataset', parallel = True)
    
    
    
if __name__ == '__main__':
    main()
    