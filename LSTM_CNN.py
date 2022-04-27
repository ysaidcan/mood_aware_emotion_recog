import numpy
import tensorflow.keras.losses
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Conv1D
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

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time
  
  
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

  
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
  
  
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
  

def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tensorflow.expand_dims(series, axis=-1)
    ds = tensorflow.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
  

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

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




EPOCHS = 1000
BATCH_SIZE = 200
TIME_STEPS =120
X_train = pd.read_csv('one_sec_sorted.csv')
X_train=X_train.drop(X_train.query('SessionLabel == 0').sample(frac=.67).index)




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
    
    window_size = 120
    train_set = seq2seq_window_dataset(scaled, window_size,batch_size=128)
    valid_set = seq2seq_window_dataset(x_test, window_size,batch_size=128)

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


    def make_model(metrics=METRICS, output_bias=None):
        model = keras.models.Sequential([Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]), LSTM(32, return_sequences=True), Dropout(0.1),LSTM(32,  return_sequences=True),Dropout(0.1) ,Dense(100,activation='relu'), Dense(100,activation='relu'), Dense(1,activation='sigmoid')])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-6),
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

    history = lstm_model.fit(train_set, epochs=EPOCHS,validation_data=valid_set)
                    
    plot_metrics(history)

    scores = lstm_model.evaluate(valid_set)

    test_predictions_baseline = lstm_model.predict(valid_set, batch_size=BATCH_SIZE)

    for name, value in zip(lstm_model.metrics_names, scores):
        print(name, ': ', value)
    print()

    #plot_cm(trim_dataset(y_temp, BATCH_SIZE), test_predictions_baseline)

