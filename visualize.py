import argparse
import dill
import os
import tensorflow as tf

from cbow import CBOW
from tensorflow.contrib.tensorboard.plugins import projector

###############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Visualize word vectors with tensorboard')
    parser.add_argument('--init_vecs', required=True, dest='init_vecs')
    parser.add_argument('--model_ckpt', required=True, dest='model_ckpt')
    parser.add_argument('--num_embed', required=True, type=int, dest='num_embed')
    parser.add_argument('--embed_metadata', required=True, dest='metadata')
    parser.add_argument('--graph_folder', required=True, dest='graph_folder')
    args = parser.parse_args()
    
    # Initialize model
    word_vectors = dill.load(open(args.init_vecs,'rb'))
    model = CBOW(len(word_vectors), word_vectors, 5e-4)
    
    # Load model of last epoch
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, args.model_ckpt)
        # Visualizing embeddings
        final_embed = sess.run(model.init_vecs)
        embedding_var = tf.Variable(final_embed[:args.num_embed], name='embedding')
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(args.graph_folder)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(args.graph_folder, args.metadata)
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, os.path.join(args.graph_folder, 'projector.ckpt'))