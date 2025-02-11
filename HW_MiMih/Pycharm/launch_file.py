import pandas as pd
from sklearn.pipeline import Pipeline
import warnings
from custom_catboost_regress import CustomCatBoostRegressor
from custom_catboost_classifier import CustomCatBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from metrics import result
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# Регрессия
# метод подготовки данных: удаление столбцов и строк с nan значениями при их наличии
def preprocess_regress(df):
    bad_columns = ['time_to_under', 'id', 'label', 'ticket_id', 'entrance_id', 'station_id', 'line_id']
    df = df.drop(columns=[x for x in bad_columns if x in df.columns])
    return df


# Классификация
# метод подготовки данных: удаление столбцов и строк с nan значениями при их наличии
def preprocess_multiclass(df):
    bad_columns = ['time_to_under', 'pass_dttm', 'id', 'ticket_id', 'entrance_id', 'station_id', 'line_id']
    df = df.drop(columns=[x for x in bad_columns if x in df.columns])
    return df


# метод получения временных признаков
def get_time_features(df):
    df.pass_dttm = pd.to_datetime(df.pass_dttm)
    df['day'] = df.pass_dttm.dt.dayofweek  # день недели
    df['hour'] = df.pass_dttm.dt.hour  # час в формате 24
    # разбиваем на промежутки активности пользования метро в течение дня,
    df['shift'] = df['hour'].apply(lambda x: 0 if 10 <= x <= 17 else (
        1 if 0 <= x <= 6 else (2 if 7 <= x <= 9 else (3 if x >= 18 else x))))
    df['workday'] = df['day'].apply(lambda x: 0 if x == 5 or x == 6 else 1)

    df = df.drop(columns=['pass_dttm'])
    return df


def main():
    df = pd.read_csv('train_dataset_train.csv', sep=',')

    df.dropna(inplace=True)
    df = df.loc[df.time_to_under > 0]

    x_regress_train, x_regress_test, y_regress_train, y_regress_test = train_test_split(
        df.drop(columns=['time_to_under']), df[['time_to_under']], test_size=0.3)

    pipe_regression = Pipeline(steps=[('preprocess', FunctionTransformer(preprocess_regress)),
                                      ('time_features', FunctionTransformer(get_time_features)),
                                      ('model', CustomCatBoostRegressor(10,1))])
    pipe_regression.fit(x_regress_train, y_regress_train)
    forecast_regress = pipe_regression.predict(x_regress_test)

    x_multiclass_train, x_multiclass_test, y_multiclass_train, y_multiclass_test = train_test_split(
        df.drop(columns=['label'])[:1000], df[['label']][:1000], test_size=0.3, random_state=0)

    pipe_multiclass = Pipeline(steps=[('preprocess', FunctionTransformer(preprocess_multiclass)),
                                      ('model', CustomCatBoostClassifier(100))])
    pipe_multiclass.fit(x_multiclass_train, y_multiclass_train)
    forecast_class = pipe_multiclass.predict(x_multiclass_test)

    final = result(y_multiclass_test, forecast_class, y_regress_test, forecast_regress)
    print(final)


if __name__ == '__main__':
    main()
