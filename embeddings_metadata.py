import argparse
import dill

###############################################################################

def create_embeddings_metadata(path2dict, save_path, num_words):
    idx2word = dill.load(open(path2dict, 'rb'))
    with open(save_path, 'w') as f:
        f.write('Index\tWord\n')
        if num_words == -1:
            for idx in range(len(idx2word)):
                f.write(str(idx) + '\t' + idx2word[idx] + '\n')
        else:
            for idx in range(num_words):
                f.write(str(idx) + '\t' + idx2word[idx] + '\n')

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read embeddings dict and create metadata for the embeddings visualization')
    parser.add_argument('--idx2word', required=True, dest='idx2word')
    parser.add_argument('--savepath', required=True, dest='save_path')
    parser.add_argument('--num_words', required=False, type=int, default=-1, dest='num_words')
    args = parser.parse_args()
    
    create_embeddings_metadata(path2dict=args.idx2word, save_path=args.save_path, num_words=args.num_words)