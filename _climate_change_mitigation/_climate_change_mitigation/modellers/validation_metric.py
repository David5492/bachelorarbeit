import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# def MAPE as additional evaluation metric:

def mean_absolute_percentage_error(y_test, y_pred): 
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# Output als Serie gestalten:

def total_validation(y_test, y_pred):
    """
    returns an output df, that has to be assigned to some variable.
    df contains some regression validation metrics.
    """
    metric_name = ['MAE', 'MAPE (%)', 'MSE', 'RMSE (Units)', 'r-squared (%)']
    metric_values = [
        round( mean_absolute_error(y_test, y_pred), 2),
        round( mean_absolute_percentage_error(y_test, y_pred), 2),
        round( mean_squared_error(y_test, y_pred), 2),
        round( np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        round( r2_score(y_test, y_pred) * 100, 2)
    ]
    df_output = pd.DataFrame(metric_values, index=metric_name, columns = ['value'], dtype='float')
    return(df_output)


