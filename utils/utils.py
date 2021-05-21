# import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

# some basic variables for testing
df_long = pd.read_csv('./data/dataframe_long.csv')
df_short = pd.read_csv('./data/dataframe_short.csv')
trials = [df_long[df_long.indTrial == t] for t in np.unique(df_long.indTrial)]
samples = np.unique(df_long.sampleID)


def split_df(df, split_col, split_vals):
    """
    takes a data frame and splits is according to the values in the specified column, the number of returned data frames
    will be equal to the number of passed split values
    :param df: the data frame that shall be splitted
    :param split_col: defines the column by which the dataframe should be splitted
    :param split_vals: defines the values that will go in each df. needs to be categorical.
    :return: a list of the generated datasets
    """

    return [df[df[split_col] == n] for n in split_vals]


def get_pH(max_sample, min_sample, attacker, width):
    """
    Normalizes the upper and lower bounds to compute how likely the attacker is to hit the goal, given the distance
    between attacker and all known points of the goal

    input:
    :param max_sample: the largest sampled y position (in dva)
    :param min_sample: the smallest sampled y position (in dva)
    :param attacker: the y position of the attacker (in dva)
    :param width: the span that is covered by the full goal (in dva)

    other units are possible, but they need to be identical for all input parameters.

    :return p_in: the probability that the attacker is inside the goal (value between 0 and 1)

    """

    # part 1: normalize the values to be aligned with the mean of the known goal, and to have positive values
    # if upper known equals lower known, everything will be normalized to the position of the sample
    mean_sample = (max_sample + min_sample) / 2

    # this is equal to the lower normalized goal
    upper_norm = abs(max_sample - mean_sample)
    attacker_norm = abs(attacker - mean_sample)

    # part 2: compute the covered and the unknown size of the goal
    covered_width = abs(max_sample - min_sample)
    free_width = width - covered_width

    # part 3: compute the probability that the attacker hits inside the goal
    p_in = 1 - stats.uniform.cdf(attacker_norm, loc=upper_norm, scale=free_width)

    return p_in


def strategy_single(sample: float, attacker: float, width=8):
    """
    computes the probability of a hit based on a single sample and returns if the value is above or below 0.5
    :param sample: the position of a single sample (float)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal, defaults to 8
    :return: the response (1 for go, 0 for no-go)
    """
    return int(get_pH(sample, sample, attacker, width) > 0.5)


def strategy_mean(samples, attacker: float, width=8):
    """
    computes the probability of a hit with mean samples and returns if the value is above or below 0.5
    :param samples: the positions of all observed samples (np array)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal (defaults to 8)
    :return: the response (1 for go, 0 for no-go)
    """
    pH_all = [get_pH(s, s, attacker, width) for s in samples]
    return int(np.mean(pH_all) > 0.5)


def strategy_accumulated(samples, attacker, width=8):
    """
    computes the probability of a hit with accumulated samples and returns if the value is above or below 0.5
    :param samples: the positions of all observed samples (np array)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal (defaults to 8)
    :return: the response (1 for go, 0 for no-go)
    """
    return int(get_pH(min(samples), max(samples), attacker, width) > 0.5)


def predict_responses(trials, strategy: str, samples: int):
    """
    returns an array with the predicted response category, based on the defined strategy
    :param trials: list with the data from all trials
    :param strategy: a string indicating the strategy used to make a decision
    :param samples: the number of samples taken into account for the response
    :return: array of responses (1 for go, 0 for no-go) with the same length as "trials"
    """

    # extract the sample position relative to the attacker
    sample_positions = [t.samplePosDegAtt.values[:samples] for t in trials]
    # since the sample is relative to the attacker, the attacker position is "0"
    attacker_position = 0

    # choose the correct strategy and get the responses
    if strategy == 'individual' or strategy == 'i':
        return [strategy_single(s[-1], attacker_position) for s in sample_positions]
    elif strategy == 'mean' or strategy == 'm':
        return [strategy_mean(s, attacker_position) for s in sample_positions]
    elif strategy == 'accumulated' or strategy == 'a':
        return [strategy_accumulated(s, attacker_position) for s in sample_positions]
    elif strategy == 'all':
        return [[strategy_single(s[-1], attacker_position) for s in sample_positions],
                [strategy_mean(s, attacker_position) for s in sample_positions],
                [strategy_accumulated(s, attacker_position) for s in sample_positions]]
    else:
        raise AttributeError(f"{strategy} is invalid. Please use 'individual' | 'i', 'mean'|'m', 'accumulated'|'a' or 'all'")


def get_performance(responses, conditions):
    """
    computes the proportion correct answers based on the response and the ground truth
    :param responses: an array that holds all the responses (0 no-go, 1 go)
    :param conditions: an array that holds the corresponding ground truth (0 pass, 1 hit )
    :return:
    """

    if len(responses) == len(conditions):
        response_correct = 1-abs(responses - conditions)
        return np.mean(response_correct)
    else:
        raise ValueError(f"responses and condition need to be of the same length but are of length {len(responses)} "
                         f"and {len(conditions)}")


def normalize_df(dataframe, columns):
    """
    normalized all columns passed to zero mean and unit variance, returns a full data set
    :param dataframe: the dataframe to normalize
    :param columns: all columns in the df that should be normalized
    :return: the data, centered around 0 and divided by it's standard deviation
    """

    for column in columns:

        data = dataframe.loc[:, column].values

        sd = np.std(data)
        mean = np.mean(data)

        dataframe.loc[:, column] = (data - mean) / sd

    return dataframe


def predict_lm_response(estimates, predictor, data, sigmoid=True):
    """
    Gets a response prediction for the linear regression model for response types
    :param estimates: the coefficients estimated by the model
    :param predictor: the column name that will be used to compute the prediction
    :param data:  the data frame
    :param sigmoid:  if the prediction should additionally be passed through a sigmoid function (normalized between 0 and 1)
    :return: the predicted response probability (either as sigmoid or as ln odds)
    """
    b_0 = estimates.loc['(Intercept)', 'Estimate']
    b_1 = estimates.loc[predictor, 'Estimate']
    b_2 = estimates.loc['hitGoal', 'Estimate']

    ln_odds = b_0 + b_1 * data[predictor] + b_2 * data['hitGoal']
    if sigmoid:
        prediction = 1 / (1 + np.exp(-ln_odds.values))
    else:
        prediction = ln_odds

    return prediction


def get_filled_vec(data, fillcol, group):
    """
    Fills the nas in the data frame with the group means
    :param data: the dataframe
    :param fillcol: the column that contains nas
    :param group: the groups for which the means should be computed
    :return: the filled column
    """

    for g in np.unique(data[group]):
        g_rt = np.nanmean(data.loc[data[group] == g, fillcol])
        idx = np.where((data[group] == g) & (data['goResp'] == 0))[0]

        data.loc[idx, fillcol] = g_rt

    return data[fillcol]


def get_distance(data, columns, relative_to):
    """
    computes the absolute absolute distance between two values and normalizes such that a 0-1 distribution has a mean of 0
    and ranges between -1 and 1
    :param data: the data frame to be manipulated
    :param columns: the columns to be manipulated
    :param relative_to: the column that holds the value relative tow which the distance will be computed
    :return: the modified data frame
    """
    for column in columns:
        data.loc[:, column] = 2 * (0.5 - abs(data[relative_to] - data[column]))

    return data
