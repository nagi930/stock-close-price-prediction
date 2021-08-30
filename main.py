from FinanceDataReader import DataReader
from preprocess import *
from indicator import *
from dataset import *

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

SHUFFLE_SIZE = 50
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 8
HISTORY_SIZE = 20


def main(stock_code, start, end):
    # get stock ohlcv
    datasets = DataReader(stock_code, start, end)

    # preprocessing
    pre = Preprocess(datasets)

    pre.add_indicator(Rsi, 5)
    pre.add_indicator(Rsi, 10)
    pre.add_indicator(Stochastic, 5, 3, False)
    pre.add_indicator(Stochastic, 10, 3, False)
    pre.add_indicator(Stochastic, 5, 3, True)
    pre.add_indicator(Stochastic, 10, 3, True)
    pre.add_indicator(MeanAverage, 5)
    pre.add_indicator(MeanAverage, 10)
    pre.add_indicator(Estrangement, 5)
    pre.add_indicator(Estrangement, 10)

    pre.dropna(inplace=True)

    # scaling
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    dataset = X_scaler.fit_transform(pre.df)
    label = y_scaler.fit_transform(np.array(pre.df.Close).reshape((-1, 1)))

    test_start_index = train_end_index = int(len(dataset) * 0.8)

    # create dataset object
    X_train, y_train = multi_step(dataset, label, 0, train_end_index, HISTORY_SIZE, 5, 1)
    X_test, y_test = multi_step(dataset, label, test_start_index, None, HISTORY_SIZE, 5, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache().shuffle(SHUFFLE_SIZE).batch(TRAIN_BATCH_SIZE).repeat().prefetch(1)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE).repeat()

    # create LSTM model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=X_train.shape[-2:]))
    model.add(tf.keras.layers.LSTM(16, activation='relu'))
    model.add(tf.keras.layers.Dense(5))
    model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    # model fit
    model.fit(train_dataset, epochs=50, steps_per_epoch=50,
              validation_data=test_dataset, validation_steps=5, callbacks=[es, mc])

    # model predict
    predict = model.predict(np.array(dataset[-HISTORY_SIZE:, :]).reshape((1, HISTORY_SIZE, dataset.shape[1])))
    predict = y_scaler.inverse_transform(predict.reshape((-1, 1))).flatten()

    return predict


if __name__ == '__main__':
    df = pd.read_csv('Stock_List.csv', dtype={'종목코드': str})
    start, end = '20200101', '20210808'

    result = []
    predict_days = ['2021-08-09', '2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13']

    for idx, code in enumerate(df['종목코드'], 1):
        predicted = main(code, start, end)
        prediction = {'stock_code': code}

        for day, p in zip(predict_days, predicted):
            prediction[day] = p

        result.append(prediction)

        print(idx, '/', len(df['종목코드']))

    print(result)

