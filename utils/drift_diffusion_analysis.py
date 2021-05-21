import pandas as pd
import numpy as np
import pickle
import os

from ddm import Fittable, Model, Sample, Bound
from ddm.models import LossRobustBIC, DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, Drift
from ddm.functions import fit_adjust_model, display_model

path_models = './models/'


def get_ddm_sample(data, subject, correct_name):
    """
    reduces the data frame and returns a ddm sample
    :param data: the data frame
    :param subject: the subject we want to fit
    :param correct_name: the name of the column that holds the answer values
    :return: the sample
    """
    # define the sample for the models
    ddm_df = data

    # reduce to one subject
    ddm_df = ddm_df[ddm_df.subject == subject]

    # reduce my data file to the necessary columns
    ddm_df = ddm_df.loc[:, ['rea_time', 'goResp', 'answer', 'sampleProbHit_01', 'sampleProbHit_02', 'sampleProbHit_03',
                            'sampleProbHit_04', 'sampleProbHit_05', 'sampleProbHit_06']]

    # drop all rows that contain nans and reset the index
    ddm_df.dropna(axis=0, inplace=True)
    ddm_df.reset_index(drop=True, inplace=True)

    # turn my datafile into a pyDDM sample
    sample = Sample.from_pandas_dataframe(ddm_df, rt_column_name="rea_time", correct_column_name=correct_name)

    return sample


# step 1: define a collapsing boundary class.
# this class definition was taken from https://pyddm.readthedocs.io/en/latest/cookbook/bounds.html#bound-exp-delay
class BoundCollapsingExponentialDelay(Bound):
    """Bound collapses exponentially over time.

    Takes three parameters:

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins, in seconds
    """
    name = "Delayed exponential collapsing bound"
    required_parameters = ["B", "tau", "t1"]

    def get_bound(self, t, conditions, **kwargs):
        if t <= self.t1:
            return self.B
        if t > self.t1:
            return self.B * np.exp(-self.tau * (t - self.t1))


# step 2 define the drift classes

# the drift for the fist model
class FirstValDrift(Drift):
    """ returns the first evidence value multiplied by the scale

    Takes one parameter:

    'scale' - scales the evidence up or down to the final drift rate

    """
    name = "returns drift of first value"
    required_conditions = ["sampleProbHit_01"]
    required_parameters = ["scale"]

    def get_drift(self, t, conditions, **kwargs):
        return conditions['sampleProbHit_01'] * self.scale


# the drift for the second model

class ThreshDrift(Drift):
    """ returns 0 till a threshold is crossed. Then, returns the first evidence that was above the threshold

    Takes two parameters:

    'scale' - scales the evidence up or down to the final drift rate
    'thresh' - the threshold up to which the model returns "0"

    """
    name = "drifts with the first value above threshold"
    required_conditions = ["sampleProbHit_01", "sampleProbHit_02", "sampleProbHit_03", "sampleProbHit_04",
                           "sampleProbHit_05", "sampleProbHit_06"]
    required_parameters = ["scale", "thresh"]
    # set a drift value here to access it later
    drift_value = 0
    # this schema defines the temporal structure how samples appeared
    time_schema = np.linspace(0, 0.88, 6)

    def get_drift(self, t, conditions, **kwargs):
        # get all samples that were already shown
        passed = self.time_schema[(self.time_schema - t) <= 0]
        # get the most recent of these
        prob = self.required_conditions[np.argmax(passed)]
        # check if we have to set a drift value
        if (conditions[prob] > self.thresh) and (self.drift_value == 0):
            # set the drift value
            ThreshDrift.drift_value = conditions[prob] * self.scale
        # return the drift value
        return self.drift_value


# Define the Drift for the Third Model
class ContinuousUpdate(Drift):
    """ always returns the current evidence * scale

    Takes one parameter:

    'scale' - scales the evidence up or down to the final drift rate

    """
    name = "continuously updating drifts"
    required_conditions = ["sampleProbHit_01", "sampleProbHit_02", "sampleProbHit_03", "sampleProbHit_04",
                           "sampleProbHit_05", "sampleProbHit_06"]
    required_parameters = ["scale"]
    # this schema defines the temporal structure how samples appeared
    time_schema = np.linspace(0, 0.88, 6)

    def get_drift(self, t, conditions, **kwargs):
        # get all samples that were already shown
        passed = self.time_schema[(self.time_schema - t) <= 0]
        # get the most recent
        prob = self.required_conditions[np.argmax(passed)]
        # multiply with scale and return
        return conditions[prob] * self.scale


# define all 3 models
# model one: immediate constant drift

ddm1 = Model(name='drift rate depends on first tw (fitted)',
             # custom, fittable drift rate
             drift=FirstValDrift(scale=Fittable(minval=0.1, maxval=1)),
             # constant, fittable noise
             noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)),
             # custom, fittable boundary
             bound=BoundCollapsingExponentialDelay(B=1,
                                                   tau=Fittable(minval=3, maxval=7),
                                                   t1=Fittable(minval=0.5, maxval=1.5)),
             # constant, fittable non-decision time
             overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)),
             dx=.001, dt=.01, T_dur=1)

# model two: delayed drift with threshold
ddm2 = Model(name='drift starts after threshold was crossed',
             # custom, fittable drift rate
             drift=ThreshDrift(scale=Fittable(minval=1, maxval=10), thresh=Fittable(minval=0.1, maxval=1)),
             noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)),
             # custom, fittable boundary
             bound=BoundCollapsingExponentialDelay(B=1,
                                                   tau=Fittable(minval=0.1, maxval=5),
                                                   t1=Fittable(minval=0, maxval=1)),
             # constant, fittable non-decision time
             overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)),
             dx=.001, dt=.01, T_dur=1)

# model three: drift according to the the lastest sample
ddm3 = Model(name='drift changes with every new sample',
             # custom, fittable drift rate
             drift=ContinuousUpdate(scale=Fittable(minval=1, maxval=10)),
             # constant, fittable noise
             noise=NoiseConstant(noise=Fittable(minval=.5, maxval=4)),
             # custom, fittable boundary
             bound=BoundCollapsingExponentialDelay(B=1,
                                                   tau=Fittable(minval=0.1, maxval=5),
                                                   t1=Fittable(minval=0, maxval=1)),
             # constant, fittable non-decision time
             overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1)),
             dx=.001, dt=.01, T_dur=1)


def fit_save_ddm(sample, ddm, name):
    """
    checks if a ddm was already fitted, loads if yes, fits new one if not
    :param sample: the sample generated with get_ddm_sample
    :param ddm: the name of the ddm
    :param name: the filename
    :return: the fitted model
    """

    if os.path.exists(path_models + name):
        print('model was fitted, returning fitted model')
        with open(path_models + name, 'rb') as file:
            model = pickle.load(file)
    else:

        if ddm == 'ddm1':
            model = ddm1
        elif ddm == 'ddm2':
            model = ddm2
        elif ddm == 'ddm3':
            model = ddm3
        else:
            raise ValueError('only ddm1, ddm2 and ddm3 are valid')

        print('Fitting a new model. this will take some time')
        fit_adjust_model(sample, model, lossfunction=LossRobustBIC, verbose=False)
        with open(path_models + name, 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    return model


def get_ddm_responses(ddm, data, subject):
    """
    generate responses to dataset based on a fitted ddm
    :param ddm: the fitted ddm
    :param data: the data to which we want to create responses
    :param subject: the subject for which we want to subset data
    :return: the data frame with predicted response columns
    """

    obs_file = data[data.subject == subject]
    model_pred = obs_file.copy()

    # solve the models for every trial, get an estimate of the response time, and of a true/false answer
    # finally, get the response by comparing true/false to the condition
    for r in obs_file.index:
        conditions = {
            'sampleProbHit_01': obs_file.loc[r, 'sampleProbHit_01'],
            'sampleProbHit_02': obs_file.loc[r, 'sampleProbHit_02'],
            'sampleProbHit_03': obs_file.loc[r, 'sampleProbHit_03'],
            'sampleProbHit_04': obs_file.loc[r, 'sampleProbHit_04'],
            'sampleProbHit_05': obs_file.loc[r, 'sampleProbHit_05'],
            'sampleProbHit_06': obs_file.loc[r, 'sampleProbHit_06'],

        }

        Solution = ddm.solve(conditions=conditions)

        model_pred.loc[r, 'rea_time'] = Solution.resample(1)
        model_pred.loc[r, 'probAnswer'] = Solution.prob_correct()

    model_pred.loc[:, 'answer'] = np.round(model_pred.probAnswer)
    model_pred.loc[:, 'goResp'] = 1 - (abs(model_pred.answer - model_pred.hitGoal))

    return model_pred


def match_response_types(model_data):
    """
    matches correct/incorrect responses to the ground truth condition of a trial
    :param model_data: the data with model-generated response predictions
    :return:
    """
    for ix in model_data.index:
        if model_data.loc[ix, 'goResp'] == 1:
            if model_data.loc[ix, 'hitGoal'] == 1:
                model_data.loc[ix, 'response_cat'] = 'HIT'
            else:
                model_data.loc[ix, 'response_cat'] = 'FALSE ALARM'
        else:
            if model_data.loc[ix, 'hitGoal'] == 1:
                model_data.loc[ix, 'response_cat'] = 'MISS'
            else:
                model_data.loc[ix, 'response_cat'] = 'CORRECT REJECTION'

        return model_data
