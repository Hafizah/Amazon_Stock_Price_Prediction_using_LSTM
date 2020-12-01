![Banner](https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/Banner.png)

<p align="center">
  <img width="700" height="350" src="https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/Header.jpg">
</p>

**Introduction:**<br>
One might wish that the time machine in the movie "Back To The Future" really existed! Stock price prediction would have been more accurate. However, no one can predict the future!

**Objective:**<br>
Predict Amazon stock prices for the next 7 days using LSTM networks with TensorFlow.

**Why LSTM?**<br>
The picture below shows the cell structures of Long Short-Term Memory (LSTM) and Recurrent Neural Network (RNN). LSTM, which is a special type of RNN, consists of an input gate, a forget gate and an output gate. These cells can maintain information in memory for a long period of time. This ability is perfect for predicting stock prices since it can store past information and learn its pattern.

<p align="center">
  <img width="700" height="350" src="https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/lstm.jpg">
</p>

**Methodology:**
1. Import all libraries: Pandas, Numpy, Matplotlib, Sklearn, tensorflow.
2. Upload data and perform exploratory data analysis.
3. Use "split_sequence" function to split the datasets into training and test sets. The terms "n_steps_in" is the number of inputs and "n_steps_ out" is the number of outputs. In this case, n_steps_out = 7. The term "break" is used to stop the loop in case the number of sequences exceeds the maximum length. At the end, the function returns numpy arrays of values X (past closing prices) and y (future prices).
4. Construct a neural network. Use LSTM with 30 neurons in the first hidden layer and 7 neurons in the output layer for the next 7 days of price prediction.
5. Compile the model. The mean squared error is used as a loss function. Use the Adam algorithm as an optimizer.
6. Train the model. The model is fitted using 150 training epochs with a batch size of 30 training examples. The amount of epochs depends on whether cpu/gpu-hardware is used. More time is needed to train the model using a CPU. An epoch is the amount of times the entire dataset is passed forward and backward through the neural network. In this case 150 times. 1 epoch is too big to feed to the computer at once so, it is divided into several batches. Batch size = the number of training examples. Validation_split = the percentage of the training data held back to validate performance. In this case, 10 %.
7. Plot loss and accuracy patters. The plot shows that the training loss drops below the test loss and eventually converge from 40 epochs onwards. The accuracy plot shows that both training and test values start to diverge at around 100 epochs. This can be a sign of underfitting or overfitting.

<p align="center">
  <img width="700" height="350" src="https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/Loss%20and%20accuracy.jpg">
</p>

8. Model validation. A comparison is shown between prediction and actual values. Both values converge between the 5th and 6th day.

<p align="center">
  <img width="700" height="350" src="https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/Predicted%20vs%20actual.jpg">
</p>

9. Future prediction. "inverse_transform" is used to transform the normalized values back to the original values. The diagram below shows the prediction for the next 7 days.

<p align="center">
  <img width="700" height="350" src="https://github.com/Hafizah/Amazon_Stock_Price_Prediction_using_LSTM/blob/main/Images/Prediction.jpg">
</p>

**Conclusion:**
The machine learning model predicts an increase in stock prices. From the prediction above, we see that LSTM networks are able to predict future stock prices. According to Forbes.com, despite the Coronavirus pandemic, Amazon has benefited millions since people turned to online marketplaces for essential requirements.

**Improvement:**

i. Below are steps that were taken to optimize the model perfomance:
- Used different scalers: RobustScaler, MinMaxScaler
- Changed the learning rate : 0.1, 0.01, 0.001, 0.0001, 0.00001
- Doubled the amount of datasets downloaded from the Yahoo finance website
- Changed the amount of the most recent datasets : 1000, 800, 500, 250
- Used different loss functions: MSE, RMSE, MAE
- Tweaked network size layers : between 3-8 layers
- Tweaked network size neurons : 60-20
- Used different optimizers: SGD, Adam
- Used different activation functions : ReLu, Sigmoid, Tanh

ii. Steps for future improvement:
- Try the 9 steps above again using different combinations.
- Use multiple input features instead of one input feature.
- Learn more techniques and improve by reading literature on this topic.
