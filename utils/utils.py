# import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats


def split_df(df, split_col, split_vals):
    """
    :param df: the data frame that shall be splitted
    :param split_col: defines the column by which the dataframe should be splitted
    :param split_vals: defines the values that will go in each df
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
    :param sample: the position of a single sample (float)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal, defaults to 8
    :return: the response (1 for go, 0 for no-go)
    """
    return int(get_pH(sample, sample, attacker, width) > 0.5)


def strategy_mean(samples, attacker: float, width=8):
    """
    :param samples: the positions of all observed samples (np array)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal (defaults to 8)
    :return: the response (1 for go, 0 for no-go)
    """
    pH_all = [get_pH(s, s, attacker, width) for s in samples]
    return int(np.mean(pH_all) > 0.5)


def strategy_accumulated(samples, attacker, width=8):
    """
    :param samples: the positions of all observed samples (np array)
    :param attacker: the position of the attacker (float)
    :param width: the width of the goal (defaults to 8)
    :return: the response (1 for go, 0 for no-go)
    """
    return int(get_pH(min(samples), max(samples), attacker, width) > 0.5)


def predict_responses(trials, strategy: str, samples: int):
    """
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
