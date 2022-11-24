import warnings
from catboost import CatBoostClassifier

import sys

warnings.filterwarnings("ignore")


class CustomCatBoostClassifier(CatBoostClassifier):
    def __init__(self, iterations=1000):
        '''
        iterations: число деревьев
        '''
        super().__init__()
        self.y = None
        self.x = None
        self.iterations = iterations

        # метод создающий данные для валидации

    def set_eval_data(self):
        df = self.x.join(self.y)
        df = df.drop_duplicates(subset=['label'])
        return df.drop(columns=['label']), df[['label']]

    # улучшенный fit, деревьев можно задать много, но выбираем лучшее количество - не переобучимся
    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, sample_weight=None,
            baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        self.set_params(iterations=self.iterations, loss_function='MultiClass')
        self.x = X
        self.y = y
        eval_x, eval_y = self.set_eval_data()
        super().fit(self.x, self.y,
                    cat_features=['ticket_type_nm', 'station_nm', 'line_nm', 'entrance_nm'],
                    eval_set=(eval_x, eval_y),
                    use_best_model=True)
