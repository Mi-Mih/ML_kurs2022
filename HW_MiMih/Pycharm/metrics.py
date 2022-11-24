from sklearn.metrics import r2_score, recall_score

def result(actual_class, forecast_class, actual_regress, forecast_regress):
    print('R2: ',r2_score(actual_regress, forecast_regress))
    print('recall: ',recall_score(actual_class, forecast_class, average='micro'))
    return 0.5 * r2_score(actual_regress, forecast_regress) + 0.5 * recall_score(actual_class, forecast_class, average='micro')
