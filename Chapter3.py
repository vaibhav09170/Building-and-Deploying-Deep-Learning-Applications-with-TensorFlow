import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

tf.compat.v1.disable_eager_execution()
# Turn off TensorFlow warning Messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with tf.device('/CPU:0'):
    trainingDataDF = pd.read_csv("Input\sales_data_training.csv", dtype=float)
    
    print(trainingDataDF.head())
    
    XTraining=trainingDataDF.drop('total_earnings', axis=1).values
    YTraining=trainingDataDF[['total_earnings']].values
    
    testDataDF = pd.read_csv("Input\sales_data_test.csv", dtype=float)
    
    XTesting = testDataDF.drop('total_earnings', axis=1).values
    YTesting = testDataDF[['total_earnings']].values
    
    XScaler = MinMaxScaler(feature_range=(0,1))
    YScaler = MinMaxScaler(feature_range=(0,1))
    
    XScalerTraning = XScaler.fit_transform(XTraining)
    YScalerTraning = YScaler.fit_transform(YTraining)
    
    XScalerTesting = XScaler.fit_transform(XTesting)
    YscalerTesting = YScaler.fit_transform(YTesting)
    
    print(XScalerTesting.shape)
    print(YscalerTesting.shape)
    
    print("Note: Y Values were scaled by multiplying by {:.10f} and adding {:.4f}".format(YScaler.scale_[0], YScaler.min_[0]))
    
    #Define model parameters
    learning_rate = 0.001
    trainingEpochs = 100
    display_step = 5
    
    # Define how many input and output are in our nerual networks
    number_of_inputs = 9
    number_of_outputs = 1
    
    # Define how many neurons we want in each layer of our neural network
    layer1_nodes = 50
    layer2_nodes = 100
    layer3_nodes = 50
    
    # section one: Define the layers of the nerual networks
    
    # Input Layer
    with tf.compat.v1.variable_scope('input'):
        X =tf.compat.v1.placeholder(tf.float32, shape=(None,number_of_inputs)) 
        
    # Layer 1
    with tf.compat.v1.variable_scope('Layer_1'):
        weights = tf.compat.v1.get_variable(name = 'weights1', shape=[number_of_inputs,layer1_nodes], initializer=tf.keras.initializers.GlorotUniform())
        biases = tf.compat.v1.get_variable(name = 'biases1', shape=[layer1_nodes], initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)
        
    with tf.compat.v1.variable_scope('Layer_2'):
        weights = tf.compat.v1.get_variable(name = 'weights2', shape=[layer1_nodes,layer2_nodes], initializer=tf.keras.initializers.GlorotUniform())
        biases = tf.compat.v1.get_variable(name = 'biases2', shape=[layer2_nodes], initializer=tf.zeros_initializer())
        layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)
        
    with tf.compat.v1.variable_scope('Layer_3'):
        weights = tf.compat.v1.get_variable(name = 'weights3', shape=[layer2_nodes,layer3_nodes], initializer=tf.keras.initializers.GlorotUniform())
        biases = tf.compat.v1.get_variable(name = 'biases3', shape=[layer3_nodes], initializer=tf.zeros_initializer())
        layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)
        
    with tf.compat.v1.variable_scope('output'):
        weights = tf.compat.v1.get_variable(name = 'weights3', shape=[layer3_nodes,number_of_outputs], initializer=tf.keras.initializers.GlorotUniform())
        biases = tf.compat.v1.get_variable(name = 'biases3', shape=[number_of_outputs], initializer=tf.zeros_initializer())
        prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)
        
    with tf.compat.v1.variable_scope('cost'):
        Y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))
        cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(prediction,Y))

    with tf.compat.v1.variable_scope('train'):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.compat.v1.variable_scope('logging'):
        tf.compat.v1.summary.scalar('current_cost',cost)
        summary = tf.compat.v1.summary.merge_all()
        
    saver = tf.compat.v1.train.Saver() 
        
    with tf.compat.v1.Session() as session:
        #Run the global variable initializer all variable and layer
        # session.run(tf.compat.v1.global_variables_initializer())
        
        # Load the model from the disk
        
        saver.restore(session,"model/trainedModel.ckpt")
        print(f"tranined Model has been loaded from the disk ")
        
        trainingCost = session.run(cost, feed_dict={X:XScalerTraning, Y:YScalerTraning})
        testingCost = session.run(cost, feed_dict={X:XScalerTesting, Y:YscalerTesting})
        print(f" Training cost : {trainingCost} : TestingCost : {testingCost} ")
        
        #Run the optimizer over and over to train the network. 
        # one epoch is one full run through the traning data set. 
        
        # for epoch in range(trainingEpochs):
        #     # Feed in the training data and do one step of neural network training
        #     session.run(optimizer, feed_dict={X: XScalerTraning, Y: YScalerTraning})
            
        #     trainingWriter = tf.compat.v1.summary.FileWriter("Logs/traning", session.graph)
        #     testingWriter = tf.compat.v1.summary.FileWriter("Logs/testing", session.graph)
            
        #     #print(f"Training Pass : {epoch}")
            
        #     if epoch % 5 == 0:
        #         trainingCost, traningSummary = session.run([cost,summary], feed_dict={X: XScalerTraning, Y:YScalerTraning})
        #         testingCost, testingSummary = session.run([cost,summary], feed_dict={X: XScalerTesting, Y:YscalerTesting})
                
        #         print(f"{epoch} : Training cost : {trainingCost} : TestingCost : {testingCost} ")
                
        #         trainingWriter.add_summary(traningSummary, epoch)
        #         testingWriter.add_summary(testingSummary, epoch)
                
        # print(f"Training is completed!")
        
        FinaltrainingCost = session.run(cost, feed_dict={X: XScalerTraning, Y:YScalerTraning})
        FinaltestingCost = session.run(cost, feed_dict={X: XScalerTesting, Y:YscalerTesting})
                
        print(f": Final Training cost : {FinaltrainingCost} : Final TestingCost : {FinaltestingCost}")
        
        YPredictedScaled = session.run(prediction, feed_dict={X:XScalerTesting})
        yPredicted = YScaler.inverse_transform(YPredictedScaled)
        
        realEarnings = testDataDF['total_earnings'].values[0]
        predictedEarning = yPredicted[0][0]
        
        print(f" \n The actual Earning : {realEarnings}")
        print(f" \n Accuracy of our nerual network and predicted Earning is : {predictedEarning}")
        
        SavePath = saver.save(session, "model/trainedModel.ckpt") 
        print(f"model saved : {SavePath}")
        
        
        
        