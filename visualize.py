import dill
import tensorflow as tf

from cbow import CBOW
from tensorflow.contrib.tensorboard.plugins import projector

###############################################################################

if __name__ == '__main__':
    
    # Initialize model
    word_vectors = dill.load(open('word_vectors','rb'))
    model = CBOW(len(word_vectors), word_vectors, 5e-4)
    
    # Load model of last epoch
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '../word_vec_models/epoch_1/model.ckpt')
        # Visualizing embeddings
        # final_embed = dill.load(open('word_vecs','rb'))
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('final_embeddings')
        embedding = config.embeddings.add()
        embedding.tensor_name = model.init_vecs.name
        embedding.metadata_path = './embeddings_metadata.tsv'
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([model.init_vecs])
        saver_embed.save(sess, '../word_vec_graphs/projector.ckpt')
        # dill.dump(final_embed, open('word_vecs','wb'))