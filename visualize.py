import dill
import tensorflow as tf

from cbow import CBOW

###############################################################################

if __name__ == '__main__':
    
    # Initialize model
    word_vectors = dill.load(open('word_vectors','rb'))
    model = CBOW(len(word_vectors), word_vectors, 5e-4)
    
    # Load model of last epoch
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '../word_vec_models/epoch_30/model.ckpt')
        # Visualizing embeddings
        final_embed = sess.run(model.word_vecs)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('final embeddings')
        embedding = config.embeddings.add()
        embedding.tensor_name = model.word_vecs.name
        embedding.metadata_path = embed_data_path
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([model.word_vecs])
        saver_embed.save(sess, '../word_vec_graphs/')