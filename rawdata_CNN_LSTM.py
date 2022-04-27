import numpy
import tensorflow.keras.losses
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D
from keras.layers import Dropout
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix

import seaborn as sns

from keras import optimizers
from tensorflow import keras
from keras.metrics import Precision, Recall, Accuracy
from sklearn.model_selection import TimeSeriesSplit





#from imblearn.under_sampling import RandomUnderSampler, NearMiss


from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', numpy.sum(cm[1]))

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()



def build_timeseries(mat, y_col_index, TIME_STEPS):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = numpy.zeros((dim_0, TIME_STEPS, dim_1-1))
    y = numpy.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i,0:y_col_index]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y



def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat




EPOCHS = 200
BATCH_SIZE = 1000
TIME_STEPS =300
X_train = pd.read_csv('one_sec_sorted.csv')
X_train = X_train.drop(X_train.query('SessionLabel == 0').sample(frac=.67).index)



X_train['SessionLabel'] = LabelEncoder().fit_transform(X_train['SessionLabel'])
label_train=X_train['SessionLabel']

neg, pos = numpy.bincount(X_train['SessionLabel'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

#nm = NearMiss()

#rus = RandomUnderSampler(random_state=42)
#X_train, label_train = nm.fit_resample(X_train, label_train)

values = X_train.values

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=False)
cvscores = []
for train, test in kfold.split(values, label_train):
    train=train[(train < len(values))]
    test = test[(test < len(values))]
    values1= values[train]
    x_test = values[test]
    print("Train and Test size", len(values), len(x_test))

    values1 = values1.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values1)
    x_test = scaler.fit_transform(x_test)
    print("hello")
    x_t, y_t = build_timeseries(scaled, -1, TIME_STEPS)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    x_temp, y_temp = build_timeseries(x_test, -1, TIME_STEPS)
    x_val, x_test_t = numpy.split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_val, y_test_t = numpy.split(trim_dataset(y_temp, BATCH_SIZE), 2)

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

    verbose, epochs, batch_size = 0, 100, 32
    n_timesteps, n_features, n_outputs = x_t.shape[1], x_t.shape[2], 1


    def make_model(metrics=METRICS, output_bias=None):
        model=Sequential([Conv1D(filters=128, kernel_size=32, activation='relu', input_shape=(n_timesteps, n_features)),
                          Dropout(0.2),LSTM(256, return_sequences=True),Dropout(0.2), LSTM(256, return_sequences=True), MaxPooling1D(pool_size=2),
    Flatten(), Dense(100,activation='relu'), Dense(100,activation='relu'), Dense(1,activation='sigmoid')])


        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        return model


#    lstm_model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy, optimizer=nadam_v2, metrics=[Accuracy, Precision, Recall])

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    initial_bias = numpy.log([])
    lstm_model=make_model()
    lstm_model.summary()

    history = lstm_model.fit(x_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                     validation_data=(trim_dataset(x_temp, BATCH_SIZE),
                    trim_dataset(y_temp, BATCH_SIZE)))
    plot_metrics(history)

    scores = lstm_model.evaluate(trim_dataset(x_temp, BATCH_SIZE), trim_dataset(y_temp, BATCH_SIZE), verbose=0, batch_size=BATCH_SIZE)

    test_predictions_baseline = lstm_model.predict(trim_dataset(x_temp, BATCH_SIZE), batch_size=BATCH_SIZE)

    for name, value in zip(lstm_model.metrics_names, scores):
        print(name, ': ', value)
    print()

    #plot_cm(trim_dataset(y_temp, BATCH_SIZE), test_predictions_baseline)


