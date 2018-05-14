from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import heapq
import tensorflow as tf
import numpy as np

import features_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_features, train_label), (test_features, test_label) = features_data.load_data()

    # Feature columns describe how to use the input.
    transaccion_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Transaccion', vocabulary_list=["compra", "alquiler"])
    localizacion_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Localizacion', vocabulary_list=["barcelona", "gerona"])
    superficie_feature_column = tf.feature_column.numeric_column("Superficie")
    precio_compra_feature_column = tf.feature_column.numeric_column("PrecioCompra")
    precio_alquiler_feature_column = tf.feature_column.numeric_column("PrecioAlquiler")
    #precio_feature_column = tf.feature_column.numeric_column("Precio")

    my_feature_columns = [
        tf.feature_column.indicator_column(transaccion_column),
        tf.feature_column.indicator_column(localizacion_column),
        tf.feature_column.bucketized_column(source_column = superficie_feature_column,
                                            boundaries = [50, 75, 100]),
        tf.feature_column.numeric_column(key='Habitaciones'),
        tf.feature_column.numeric_column(key='Banos'),
        tf.feature_column.bucketized_column(source_column = precio_compra_feature_column,
                                            boundaries = [1, 180000, 200000, 225000, 250000, 275000, 300000]),
        tf.feature_column.bucketized_column(source_column = precio_alquiler_feature_column,
                                            boundaries = [1, 700, 1000, 1300])
        #tf.feature_column.bucketized_column(source_column = precio_feature_column,
        #                                    boundaries = [700, 1000, 1300, 180000, 200000, 225000, 250000, 275000, 300000])
    ]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],              # two hidden layers of 10 nodes each
        feature_columns=my_feature_columns,
        n_classes=12,                       # the model must choose between 12 URLs                       
        #label_vocabulary=features_data.OUTPUTS
        )
  
    # Train the Model.
    classifier.train(
        input_fn=lambda:features_data.train_input_fn(train_features, train_label, args.batch_size),
        steps=args.train_steps)
       
  
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:features_data.eval_input_fn(test_features, test_label, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
     
    # Generate predictions from the model
    #expected = ['Url 7']
    predict_x = {
        'Transaccion': ['alquiler'],
        'Localizacion': ['gerona'],
        'Superficie': [90],
        'Habitaciones': [4],
        'Banos': [2],
        'PrecioCompra': [0],
        'PrecioAlquiler': [1100]
    }
    
    predictions = classifier.predict(
        input_fn=lambda:features_data.eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))
    
    #template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    #for pred_dict, expec in zip(predictions, expected):
    #    class_id = pred_dict['class_ids'][0]
    #    probability = pred_dict['probabilities'][class_id]
    #    print(template.format(features_data.OUTPUTS[class_id], 100 * probability, expec))    

    for pred_dict in predictions:        
        print('\nTop 3 similar outputs:')
        template=('\n {} ({:.2f}%)')
        
        top_probabilities = heapq.nlargest(3, pred_dict['probabilities'])                
        
        for i in top_probabilities:
            idx=np.argwhere(pred_dict['probabilities']==i)[0][0]
            print(template.format(features_data.OUTPUTS[idx], i*100))

        """
        for index, probability in enumerate(pred_dict['probabilities']):
            print(template.format(features_data.OUTPUTS[index], probability*100))

        for index, probability in enumerate(pred_dict['probabilities']):
            print(index, probability)

        for index, probability in enumerate(top_probabilities):
            np.argwhere(pred_dict['probabilities']==i)[0][0]
            print(template.format(features_data.OUTPUTS[index], probability*100))

        #indexes = [np.where(pred_dict['probabilities']==top_probabilities[0]), np.where(pred_dict['probabilities']==top_probabilities[0])]

        x=np.argwhere(pred_dict['probabilities']==top_probabilities[0])
        y=np.argwhere(pred_dict['probabilities']==top_probabilities[1])
        z=np.argwhere(pred_dict['probabilities']==top_probabilities[2])
        idx=np.array([x[0][0],y[0][0],z[0][0]])
        #print(idx)
    
        print(features_data.OUTPUTS[x[0][0]], 100*top_probabilities[0])
        print(features_data.OUTPUTS[y[0][0]], 100*top_probabilities[1])
        print(features_data.OUTPUTS[z[0][0]], 100*top_probabilities[2])
       
        """

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

# cd C:\Users\mponsser\Desktop\ML project\TF\
# python premade_estimator.py