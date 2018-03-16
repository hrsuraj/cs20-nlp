import argparse
import dill

###############################################################################

def create_embeddings_metadata(path2dict, save_path):
    idx2word = dill.load(open(path2dict, 'rb'))
    with open(save_path, 'w') as f:
        f.write('Word\n')
        for idx in range(len(idx2word)):
            f.write(idx2word[idx]+'\n')

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read embeddings dict and create metadata for the embeddings visualization')
    parser.add_argument('--idx2word', required=True, dest='idx2word')
    parser.add_argument('--savepath', required=True, dest='save_path')
    args = parser.parse_args()
    
    create_embeddings_metadata(path2dict=args.idx2word, save_path=args.save_path)