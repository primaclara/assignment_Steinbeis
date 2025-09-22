# assignment_Steinbeis
This project explores deep neural networks using datasets like Red Wine and Spotify. I studied over/underfitting, detecting them via training vs. validation metrics, and applied fixes like dropout and batch normalization. Cross-entropy loss improved classification, while dropout boosted generalization, raising validation accuracy.
**Introduction**
Deep learning has become the dominant approach for AI in the modern world. It entails computers making highly reliable predictions and patterns from data. Neural networks can be employed to do the simple tasks, which could be predicting the popularity of songs in Spotify, predicting the compressive strength of concrete, and predict the probability or cancelling a hotel booking. However, despite this wonderful success, it is not without issues such as the training of deep networks. The primary issues are overfitting and underfitting, where overfitting is simply a case where the model memorizes the training data and underfitting is the opposite. These two factors clearly illustrate the importance of appropriate loss functions and regularization methods for models to learn to the intervention. This is where terms such as cross-entropy loss, dropout layer, batch normalization, and early stopping are thus also very important in stabilizing training and improving performance. Hence, this report delves into these techniques with illustrations from Kaggle tutorials.

**Overfitting and Underfitting in Deep Learning**
Training a deep neural network requires finding the right balance between model complexity and generalization ability. Two key problems often observed during training are overfitting and underfitting (Akgun., 2022).

Overfitting occurs when a model learns the training data too well, including noise and irrelevant details, leading to poor performance on unseen data. It is typically detected when training loss continues to decrease while validation loss begins to increase (Sharifani., et al 2023). In contrast, underfitting arises when a model is too simple to capture the underlying patterns of the data. In this case, both training and validation losses remain high, and the model fails to achieve satisfactory accuracy (Chollet., 2021).

Different strategies exist to handle these issues. To address overfitting, common techniques include EarlyStopping, Dropout layers, Batch Normalization, and adding more training data. For example, in the Spotify dataset experiment, the baseline linear model showed underfitting, with validation loss around 0.2037. By increasing the network’s depth and training with EarlyStopping (patience = 5), the model achieved a lower minimum validation loss of 0.1985, demonstrating improved fit without overfitting. On the other hand, when the model was trained too long without regularization, validation loss started diverging from training loss, which was a clear indication of overfitting (Cheng., et al 2024).

**Cross-Entropy Loss**
In deep learning, the loss function plays a critical role in guiding the optimization process. For classification problems, the most widely used loss function is cross-entropy, as it directly measures the difference between the predicted probabilities and the true class labels. The cross-entropy function penalizes predictions that are far from the actual class, pushing the model to assign higher probabilities to the correct output (Chen., et al 2025).

There are two common variants: binary cross-entropy and categorical cross-entropy. Binary cross-entropy is applied when the task involves two classes, such as predicting whether a hotel booking will be cancelled (is_canceled = 0 or 1). In this case, the model uses a sigmoid activation in the output layer to produce probabilities between 0 and 1. For multi-class tasks, categorical cross-entropy is used along with a softmax activation, which outputs a probability distribution across multiple categories (Hossen., et al 2021).

The Kaggle tutorial on hotel booking cancellations demonstrates the effectiveness of binary cross-entropy. The model was compiled using (Quaranta., et al 2021):

This setup ensured that the network optimized towards distinguishing cancellations from non-cancellations. Compared with other loss functions, cross-entropy aligns closely with probability-based predictions, making it the natural choice for classification tasks. By strongly penalizing incorrect classifications, it accelerates convergence and improves accuracy (Cheng., et al 2024).

**Dropout Layer in Neural Networks**
Regularization methods are useful in deep learning to reduce overfitting, with the Dropout layer (Lubana., et al 2021) presenting one of the best solutions. A Dropout layer was presented by Srivastava et al., which deactivates a fraction of neurons randomly at each training step. The key idea is that the model does not over rely on specific neurons, and learns more distributive it can generate robust representations (Chen., et al 2025).

The Dropout rate, which activates 0.2 to 0.5 of the neuron's output, indicates the proportion of neurons that will be deactivated. The random selection of neurons disassociates or diminishes co-adaptation during training but, during test time, all neurons are activated output and scaled to simulate the training condition altogether. The model can, therefore, take advantage of the learning objectives fostered during training but predicts using the complete network (Cheng., et al 2024).

In the Hotel Booking dataset binary classification task, dropout layers with a rate of 0.3 were added after each dense layer of size 128. Without dropout, the training accuracy was high but validation performance degraded, indicating overfitting. After introducing dropout, the validation accuracy stabilized and the validation loss decreased more steadily, showing that dropout improved the model’s ability to generalize to unseen data (Hossen., et al 2021).

**Case Studies from Tutorials**
The concepts of overfitting, underfitting, cross-entropy, dropout, and batch normalization can be better understood through the Kaggle tutorials applied on different datasets.
In the Spotify dataset, the task was to predict song popularity (a regression problem) (Pareek., et al 2022). The baseline linear model showed underfitting, with both training and validation loss remaining high (minimum validation MAE ≈ 0.2037). By increasing the number of hidden layers and using EarlyStopping, the model improved to a lower validation loss of ≈ 0.1985, demonstrating how increasing model capacity reduces underfitting while early stopping prevents overfitting (Li., et al 2023).

In the Concrete dataset, the task was to predict compressive strength. When trained without standardization, the model diverged and generated NaN losses with SGD. When Batch Normalization layers were added before each dense layer, the network stabilized and demonstrated a smoother approach towards convergence. This again demonstrated that BatchNorm handled unstandardized data and maintained numerical stability (Cheng., et al 2024).

Finally, in the Hotel Booking dataset, there was a binary classification task for predicting cancellations with binary cross-entropy loss and sigmoid output. Predictions showed that the validation accuracy was higher when both Dropout (0.3) and Batch Normalization were applied after each dense layer which also reduced overfitting. The validation curves also stabilized indicating that utilizing these regularization techniques supported generalization (Ullah., et al 2024).

**Conclusion and Recommendations**
This study explored key issues and solutions in the training of deep neural networks with particular emphasis on overfitting, underfitting, cross-entropy, dropout, and batch normalization. Experiments conducted on several datasets suggest that model performance is highly sensitive to the appropriate amount of model complexity versus regularization. For Spotify, a deeper model combined with EarlyStopping showed less underfitting, and for Concrete, Batch Normalization showed a benefit with non-standardized inputs. For Hotel Booking, as expected, binary cross-entropy loss with Dropout aided classification accuracy. Based on these findings, several guidelines are provided:

**From these results, several recommendations emerge:**

Use cross-entropy loss for classification tasks.
Apply dropout and batch normalization to reduce overfitting.
Monitor training and validation curves closely.
Employ early stopping to avoid unnecessary training epochs.
