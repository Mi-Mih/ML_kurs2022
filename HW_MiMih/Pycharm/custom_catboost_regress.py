import warnings
from catboost import CatBoostRegressor, cv, Pool
import numpy as np
import sys

warnings.filterwarnings("ignore")


class CustomCatBoostRegressor(CatBoostRegressor):
    def __init__(self, iterations=1000, cv_numb=2):
        super(CatBoostRegressor, self).__init__()
        '''
        iterations: число деревьев
        '''
        self.y = None
        self.x = None
        self.cv_numb = cv_numb
        self.iterations = iterations

    # кросс-валидация для определения оптимального числа деревьев
    def launch_cv(self, metric='MAPE'):
        cv_data = cv(params={'loss_function': metric, 'iterations': self.iterations, 'random_seed': 0, 'depth': 10},
                     pool=Pool(self.x, label=self.y,
                               cat_features=['ticket_type_nm', 'station_nm', 'line_nm', 'entrance_nm']),
                     fold_count=2, inverted=False, shuffle=True, partition_random_seed=0, stratified=False)
        self.set_params(iterations=np.argmin(cv_data['test-' + metric + '-mean']))

    # улучшенный fit, деревьев можно задать много, но выбираем лучшее количество - не переобучимся
    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None,
            sample_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        self.x = X
        self.y = y
        if self.cv_numb>1:
           self.launch_cv()
        super().fit(self.x, self.y, cat_features=['ticket_type_nm', 'station_nm', 'line_nm', 'entrance_nm'])

