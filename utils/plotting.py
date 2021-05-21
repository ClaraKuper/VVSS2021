import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import utils.utils as utils

path_figs = './figures/'

# matplotlib pretty settings
# colormaps
colormaps = {
    # subjects
    's_cm': plt.cm.get_cmap('summer'),
    # go/no-go
    'g_cm': plt.cm.get_cmap('BrBG'),
    # performance correct/incorrect
    'p_cm': plt.cm.get_cmap('RdYlGn'),
    # condition hit/pass
    'c_cm': plt.cm.get_cmap('PuOr'),
    # time continuous
    't_cm': plt.cm.get_cmap('Blues_r')
}

# fontsize
font = {'weight': 'normal',
        'size': 25}
plt.rc('font', **font)
plt.rcParams['legend.title_fontsize'] = 25
plt.rcParams['legend.fontsize'] = 25

# plot sizes:
in2cm = 1 / 2.54
fig_sizes = {
    'in2cm': in2cm,
    'height': 10 * in2cm,
    'width': 10 * in2cm,
    'gaps': 4 * in2cm
}


def make_summary_plot(data, cmaps=colormaps, sizes=fig_sizes):
    """
    Creates a large plot summarizing behaviour in the experiment
    :param data: data frame to be visualized. Needs the columns "subject", "goResp", "reaTime", "hitGoal"
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return: arrays needed to reproduce panels 2 and 3 (reaction times, performance in hit trial, performance in pass
    trials, observer array)
    """

    # initialize the figure
    fig, axs = plt.subplots(1, 3, figsize=(sizes['width'] * 3, sizes['height']))

    obs = np.unique(data.subject)

    # set labels on combined figure
    # the first panel will be a graph of the number of go vs. no-go responses
    axs[0].set_title('Response frequencies')
    # the second panel will show the distribution of response times
    axs[1].set_title('Response times')
    # the third panel will show the proportion correct trial by condition
    axs[2].set_title('Performance')

    # plot panel 1
    # group by subject and go Response
    summary_a_bSG = data.groupby(['subject', 'goResp']).describe().answer
    # retrieve the count information from the summary
    go_count = summary_a_bSG.loc[((slice(obs[0], obs[-1])), [1]), :]['count'].values
    # flip one information
    nogo_count = -1 * summary_a_bSG.loc[((slice(obs[0], obs[-1])), [0]), :]['count'].values

    # plot everything
    axs[0].bar(obs, go_count, color=cmaps['g_cm'](np.linspace(0.2, 0.8, 2)[1]), label='go')
    axs[0].bar(obs, nogo_count, color=cmaps['g_cm'](np.linspace(0.2, 0.8, 2)[0]), label='no go')

    # Use absolute value for y-ticks
    ticks = axs[0].get_yticks()
    axs[0].set_yticklabels([np.round(int(abs(tick)) / (3 * 800), 1) for tick in ticks]);

    # add labels
    axs[0].set_ylabel('proportion of responses')
    axs[0].set_xlabel('observer')
    axs[0].legend(title='response', loc='upper right')

    subjects = np.unique(data.subject)
    s_cm = cmaps['s_cm'](np.linspace(0,1,len(subjects)))

    # plot panel 2
    # collect reaction times in a list of lists
    rts = [data[data.subject == s].rea_time.values for s in np.unique(data.subject)]
    axs[1].hist(rts, stacked=True, color=s_cm, label=obs)

    for s, c in zip(subjects, s_cm):
        axs[1].axvline(x=np.mean(data[data.subject == s].rea_time), color=c)

    axs[1].legend(title='observer', loc='upper right')

    axs[1].set_xlabel('reaction time [s]')
    axs[1].set_ylabel('# responses')

    # plot panel 3
    summary_a_bSH = data.groupby(['subject', 'hitGoal']).describe().answer
    performance_hit = summary_a_bSH.loc[((slice(obs[0], obs[-1])), [1]), :]['mean'].values
    performance_pass = summary_a_bSH.loc[((slice(obs[0], obs[-1])), [0]), :]['mean'].values

    # for every subject and trial type
    axs[2].scatter(obs, performance_hit, label='hit', color=cmaps['c_cm'](np.linspace(0.2, 0.8, 2))[1])
    axs[2].scatter(obs, performance_pass, label='pass', color=cmaps['c_cm'](np.linspace(0.2, 0.8, 2))[0])
    axs[2].set_xlabel('observer')
    axs[2].set_ylabel('proportion correct')
    axs[2].legend(title='trial type', loc='lower right')

    plt.tight_layout()

    return rts, performance_hit, performance_pass, obs


def VVSS_fig1_plot(data, rts, obs, cmaps=colormaps, sizes=fig_sizes):
    """
    Creates and saves a reaction time plot
    :param data: the data frame
    :param rts: a list of reaction times for each subject
    :param obs: a list of all observers, in the same order as the reaction times
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return:
    """
    fig, axs = plt.subplots(1, 1, figsize=(sizes['width'] * 1.3, sizes['height'] * 1.3))

    s_cm = colormaps['s_cm'](np.linspace(0, 1, len(obs)))
    # make a histogram of the reaction times, stacked by subject
    axs.hist(rts, stacked=True, color=s_cm, label=obs)

    # draw a line at the mean
    for s, c in zip(obs, s_cm):
        axs.axvline(x=np.mean(data[data.subject == s].rea_time), color=c)

    # set the labels
    axs.legend(title='observer', loc='lower left')
    axs.set_xlabel('reaction time [s]')
    axs.set_ylabel('# responses')

    fig.savefig(path_figs + 'Fig1_ResponseTimes.pdf', bbox_inches='tight')

    return None


def VVSS_fig2_plot(p_hit, p_pass, obs, cmaps=colormaps, sizes=fig_sizes):
    """
    Creates and saves a performance plot
    :param p_hit: a list of performance in "hit" trials
    :param p_pass: a list of performance in "pass" trials
    :param obs: a list of observers, in the same order as p_hit and p_pass
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return:
    """
    fig, axs = plt.subplots(1, 1, figsize=(sizes['width'] * 1.3, sizes['height'] * 1.3))

    # make a scatter plot of the performance in hit and pass trials
    axs.scatter(obs, p_hit, label='hit', color=cmaps['c_cm'](np.linspace(0.2, 0.8, 2))[1])
    axs.scatter(obs, p_pass, label='pass', color=cmaps['c_cm'](np.linspace(0.2, 0.8, 2))[0])

    # set the labels and axis
    axs.set_xlabel('observer')
    axs.set_ylabel('proportion correct')
    axs.legend(title='trial type', loc='lower left')
    axs.set_ylim([0.5, 1])

    fig.savefig(path_figs + 'Fig2_Performance.pdf', bbox_inches='tight')

    return None


def show_hit_probability(data, sizes=fig_sizes):
    """
    Creates a plot that compares hit probabilities computed with individual samples and accumulated samples
    :param data: the data from which we compute probabilites
    :param sizes: a dictionary of sizes
    :return:
    """

    fig, axs = plt.subplots(1, 6, figsize=(6 * sizes['width'], sizes['height']), sharex='all', sharey='all')

    # select samples from one trial for visualization
    samples = data[data.indTrial == 9].samplePosDegAtt.values
    # set a range of possible attacker positions
    attacker_range = np.linspace(-8, 8, 100)
    # set the goal width
    goal_width = 8

    # show the probabilities with each sample:
    for s in range(0, len(samples)):
        # utils.get_pH is the function that converts distance into probabilities
        # sampleProb computes the cdf based on only one sample
        sampleProb = [utils.get_pH(samples[s], samples[s], a, goal_width) for a in attacker_range]
        # sampleAccProb computes the cdf based on all previously seen samples
        sampleAccProb = [utils.get_pH(min(samples[0: s+1]), max(samples[0: s+1]), a, goal_width) for a in attacker_range]

        # plot both probabilities
        axs[s].plot(attacker_range, sampleProb, label='individual')
        axs[s].plot(attacker_range, sampleAccProb, label='accumulated')
        axs[s].scatter(samples[s], 1, label='sample')
        axs[s].set_title(f"sample {s+1}")

    # set some titles
    axs[0].set_xlabel('attacker distance from screen center')
    axs[0].set_ylabel('p(Hit)')
    axs[5].legend(loc=(0, -1))

    return None


def show_mean_performance(long_data, short_data, sizes=fig_sizes):
    """
    Creates a plotthat compares performance of different strategies across observers
    :param long_data: the data frame in long format
    :param short_data:  the data frame in wide format
    :param sizes: a dictionary of sizes
    :return:
    """

    # collect all trials in one array
    trials = [long_data[long_data.indTrial == t] for t in np.unique(long_data.indTrial)]
    samples = np.unique(long_data.sampleID)

    # make a data frame to collect the performance
    strategy_performance = pd.DataFrame(index = ['individual', 'mean', 'accumulated'], columns= samples)

    # get the performance for 1-6 samples shown
    for sample in samples:
        responses = utils.predict_responses(trials, 'all', sample)
        strategy_performance.loc['individual',sample] = utils.get_performance(responses[0], short_data.hitGoal.values)
        strategy_performance.loc['mean', sample] = utils.get_performance(responses[1], short_data.hitGoal.values)
        strategy_performance.loc['accumulated', sample] = utils.get_performance(responses[2], short_data.hitGoal.values)

    # plot the performance for the 3 strategies
    fig, axs = plt.subplots(1,1, figsize = (2 * sizes['width'], 2 * sizes['height']), sharex = 'all', sharey = 'all')

    axs.plot(samples, strategy_performance.loc['individual',:], label = 'individual')
    axs.plot(samples, strategy_performance.loc['mean',:], label = 'mean')
    axs.plot(samples, strategy_performance.loc['accumulated',:], label = 'accumulated')

    # add scale for y axis
    axs.set_ylim(0.5,1)
    # add labels
    axs.legend(loc = (0.2,0))
    axs.set_xlabel('n samples shown')
    axs.set_ylabel('mean performance over all trials')


def visualize_response_predictors(long_data, short_data, order=2, sizes=fig_sizes):
    """
    Create a plot to show the relation of predictor variables to outcome variables
    :param long_data: the data frame in long format
    :param short_data: the data frame in short format
    :param order: the exponential for the predictor relationships
    :param sizes: a dictionary of sizes
    :return:
    """
    fig, axs = plt.subplots(1, 3, figsize=(3 * sizes['width'], sizes['height']), sharex='all', sharey='all')

    sb.regplot(x=long_data.sampleProbHit, y=long_data.goResp, order=order, ax=axs[0])
    sb.regplot(x=long_data.sampleTimeSecGo, y=long_data.goResp, order=order, ax=axs[1])
    sb.regplot(x=short_data.sampleProbHit_01, y=short_data.goResp, order=order, ax=axs[2])

    # some axis labels and titles
    axs[0].set_xlabel('normalized probability (all samples)')
    axs[1].set_xlabel('normalized time')
    axs[2].set_xlabel('normalized probability (first sample)')

    axs[0].set_ylabel('proportion "go" responses')

    return None


def get_interaction(data, columns):
    """
    Add interaction terms to data frames for model result estimation
    :param data: the data frame
    :param columns: the columns that produce the interaction
    :return:
    """
    if len(columns) > 2:
        raise ValueError('3-way-interactions are not supported by this method.')
    else:
        return data[columns[0]] * data[columns[1]]


def plot_prediction_comparison(data, model, var, order=2, sizes=fig_sizes):
    """
    creates a plot with one panel per sample, the estimates made by the model and the
    estimates made for this time window individually
    :param model: the fitted model
    :param data: the data that the model was fitted on
    :param var: the predictor variable
    :param order: the order of the exponential function
    :param sizes: a dictionary of sizes
    :return: nothing
    """

    samples = np.unique(data.sampleID)

    fig, axs = plt.subplots(1, len(samples), figsize=(sizes['width'] * len(samples), sizes['height']), sharex='all',
                            sharey='all')

    # check if the model has an interaction term and generate if needed
    for i in model.coefs.index:
        if not i in data.columns:
            if ':' in i:
                data[i] = get_interaction(data, i.split(':'))

    for s in samples:
        # subset the dataset
        dat = data[data.sampleID == s].reset_index(drop=True)
        # get a prediction of the responses
        dat['predicted'] = model.predict(dat)

        # plot on the same axis
        sb.regplot(x=dat.sampleProbHit, y=dat.predicted, order=order, ax=axs[s - 1], line_kws={'color': 'green'},
                   scatter_kws={'color': 'grey'})
        sb.regplot(x=dat.sampleProbHit, y=dat[var], order=order, ax=axs[s - 1], line_kws={'color': 'blue'},
                   scatter_kws={'color': 'grey'})

    return None


def VVSS2021_fig3_plot(data, model, sizes=fig_sizes, cmaps=colormaps):
    """
    Create and save a plot of the results of the linear response model
    :param data: the data frame
    :param model: the fitted linear regression model
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return:
    """

    fig_pass, axs_pass = plt.subplots(1, 1, figsize=(sizes['width'], sizes['height']))
    fig_hit, axs_hit = plt.subplots(1, 1, figsize=(sizes['width'], sizes['height']))
    # combine the axis in one array
    axs = [axs_pass, axs_hit]

    sampleIDs = [1, 2, 3, 4, 5, 6]
    t_cm_discrete = cmaps['t_cm'](np.linspace(0, 1, len(sampleIDs)))

    # plot data for hit/nohit
    for state in np.unique(data.hitGoal):

        # extract data from only one state
        dat = data.loc[data.hitGoal == state, :]

        # plot the data
        for estimate, c in zip(sampleIDs, t_cm_discrete):
            pred = utils.predict_lm_response(model.coefs, 'sampleProbHit_0{}'.format(estimate), dat, sigmoid=False)
            axs[state].plot(dat['sampleProbHit_0{}'.format(estimate)], pred, color=c,
                            label='sample {}'.format(estimate), linewidth=5)

    axs[0].set_ylabel('ln odds "go"')
    axs[0].set_xlabel('normalized p[H]')
    axs[0].set_ylim([-3, 3])
    axs[0].set_xlim([-1.8, 1.8])

    axs[1].set_ylabel('ln odds "go"')
    axs[1].set_xlabel('normalized p[H]')
    axs[1].set_ylim([-3, 3])
    axs[1].set_xlim([-1.8, 1.8])

    axs[1].legend(loc=(1, 0))

    fig_pass.savefig(path_figs + "Fig3_lmResponsesPass.pdf", bbox_inches='tight')
    fig_hit.savefig(path_figs + "Fig3_lmResponsesHit.pdf", bbox_inches='tight')

    return None


def VVSS2021_fig4_plot(data, model, sizes=fig_sizes, cmaps=colormaps):
    """
    Create and save a plot of the results from the linear regression reaction time model
    :param data: the data frame
    :param model: the fitted reaction time model
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return: Nothind
    """

    fig, axs = plt.subplots(1, 1, figsize=(sizes['width'], sizes['height']))

    sampleIDs = [1, 2, 3, 4, 5, 6]
    t_cm_discrete = cmaps['t_cm'](np.linspace(0, 1, len(sampleIDs)))

    for col, c in zip(sampleIDs, t_cm_discrete):
        tw = 'sampleProbHit_0{}'.format(col)
        pred_rt = model.coefs.loc['(Intercept)', 'Estimate'] + model.coefs.loc[tw, 'Estimate'] * data.loc[:, tw]
        axs.plot(data[tw], pred_rt, label='sample {}'.format(col), color=c, linewidth=5)

    axs.legend(loc=(1, 0))
    axs.set_ylabel('response time [s]')
    axs.set_xlabel('normalized p[H]')

    fig.savefig(path_figs + "Fig4_lmRTs.pdf", bbox_inches='tight')

    return None


def illustrate_update_response(data, npbin, sizes=fig_sizes, cmaps=colormaps):
    """
    Create a figure that illustrates our assumptions how updates in the responses wold look like
    :param data: the data that generated the response
    :param npbin: the number of bins for the probability transitions
    :param sizes: a dictionary of sizes
    :return:
    """

    # model for visualization:
    fig, axs = plt.subplots(1, 3, figsize=(sizes['width'] * 3, sizes['height']),
                            sharex='all', sharey='all')

    window_prob = np.linspace(min(data.sampleProbHit), max(data.sampleProbHit), npbin)

    # take one matrix and replace all values with nans
    sample_mat = pd.DataFrame(np.zeros([npbin, npbin]), columns=window_prob, index=window_prob)
    early_mat = sample_mat.copy()
    late_mat = sample_mat.copy()

    # assume threshold
    for c in sample_mat.columns:

        for r in sample_mat.index:
            early_mat.loc[r, c] = 1 / (1 + np.exp(-r))
            late_mat.loc[r, c] = 1 / (1 + np.exp(-c))

    axs[0].pcolormesh(early_mat, cmap=cmaps['g_cm'])
    axs[1].pcolormesh((late_mat + early_mat) / 2, cmap=cmaps['g_cm'])
    axs[2].pcolormesh(late_mat, cmap=cmaps['g_cm'])

    axs[0].set_title('Old')
    axs[1].set_title('Average')
    axs[2].set_title('New')

    axs[0].set_xticks(np.arange(0, 10, 2))
    axs[0].set_yticks(np.arange(0, 10, 2))
    axs[0].set_xticklabels(np.round(early_mat.columns, 2)[::2])
    axs[0].set_yticklabels(np.round(early_mat.columns, 2)[::2])

    return early_mat, late_mat


def make_update_response_plot(data, ntbin, npbin, save=True, sizes=fig_sizes, cmaps=colormaps):
    """
    Creates and saves a plot the visualized how responses change with a given probability update
    :param data: the data frame
    :param ntbin: the number of time bins
    :param npbin: the number of probability bins
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :param save: if the image should be saved or not
    :return: nothing
    """

    # set up arrays for time and probability binning
    window_prob = np.linspace(min(data.sampleProbHit), max(data.sampleProbHit), npbin)

    window_forward = np.linspace(min(data.sampleTimeSecGo), max(data.sampleTimeSecGo), ntbin)
    window_backward = np.linspace(min(data.sampleTimeSecResp), max(data.sampleTimeSecResp), ntbin)

    # initialize the figure
    fig, axs = plt.subplots(2, ntbin - 1, figsize=((ntbin - 1) * sizes['width'], 2 * sizes['height']), 
                                          sharex='all', sharey='all')

    # initialize lists to collect data
    changes_list_fw = []
    changes_list_bw = []

    # make one data frame per time window
    for tw in range(0, ntbin - 2):

        # intitialize the data frames
        change_fw = pd.DataFrame()
        change_bw = pd.DataFrame()

        # filter data from this tw
        ctw_fw = data.loc[
            np.where((data.sampleTimeSecGo >= window_forward[tw]) & (data.sampleTimeSecGo < window_forward[tw + 1]))]
        ctw_bw = data.loc[np.where(
            (data.sampleTimeSecResp >= window_backward[tw]) & (data.sampleTimeSecResp < window_backward[tw + 1]))]

        # filter data from the next tw
        ntw_fw = data.loc[np.where(
            (data.sampleTimeSecGo >= window_forward[tw + 1]) & (data.sampleTimeSecGo < window_forward[tw + 2]))]
        ntw_bw = data.loc[np.where(
            (data.sampleTimeSecResp >= window_backward[tw + 1]) & (data.sampleTimeSecResp < window_backward[tw + 2]))]

        # go through all probabilities in tw 1
        for p_start in range(0, npbin - 1):

            pst_low = window_prob[p_start]
            pst_up = window_prob[p_start + 1]

            # get all IDs that match the time condition 
            start_IDs_fw = \
                ctw_fw.iloc[np.where((ctw_fw.sampleAccprobHit >= pst_low) & (ctw_fw.sampleAccprobHit < pst_up))[0]][
                    'indTrial']
            start_IDs_bw = \
                ctw_bw.iloc[np.where((ctw_bw.sampleAccprobHit >= pst_low) & (ctw_bw.sampleAccprobHit < pst_up))[0]][
                    'indTrial']

            # go through all probabilities in the next time window
            for p_end in range(0, npbin - 1):
                pend_low = window_prob[p_end]
                pend_up = window_prob[p_end + 1]

                # get all IDs that match the probability condition
                end_IDs_fw = \
                    ntw_fw.iloc[
                        np.where((ntw_fw.sampleAccprobHit >= pend_low) & (ntw_fw.sampleAccprobHit < pend_up))[0]][
                        'indTrial']
                end_IDs_bw = \
                    ntw_bw.iloc[
                        np.where((ntw_bw.sampleAccprobHit >= pend_low) & (ntw_bw.sampleAccprobHit < pend_up))[0]][
                        'indTrial']

                # get the intercept between the two lists
                ID_list_fw = np.intersect1d(start_IDs_fw, end_IDs_fw)
                ID_list_bw = np.intersect1d(start_IDs_bw, end_IDs_bw)

                # filter the data
                dat_fw = data.loc[np.where(data.indTrial.isin(ID_list_fw))]
                # filter backwards data
                dat_bw = data.loc[np.where(data.indTrial.isin(ID_list_bw))]

                # get the mean response in this data
                change_fw.loc[window_prob[p_start], window_prob[p_end]] = np.mean(dat_fw.goResp)
                change_bw.loc[window_prob[p_start], window_prob[p_end]] = np.mean(dat_bw.goResp)

        # save all computed data frames in one list
        changes_list_fw.append(change_fw)
        changes_list_bw.append(change_bw)

        # plot
        p1 = axs[0, tw].pcolormesh(change_fw, cmap=cmaps['g_cm'])
        axs[0, tw].set_title(
            '{} to {}[s]'.format(np.round(window_backward[tw], 2), np.round(window_backward[tw + 1], 2)))
        axs[1, tw].set_title(
            '{} to {}[s]'.format(np.round(window_forward[tw], 2), np.round(window_forward[tw + 1], 2)))
        axs[0, tw].set_xticks(np.linspace(0, 10, 9)[::2])
        axs[0, tw].set_yticklabels(np.round(np.linspace(0, 1, 9), 1)[::2])
        axs[0, tw].set_yticks(np.linspace(0, 10, 9)[::2])
        axs[0, tw].set_xticklabels(np.round(np.linspace(0, 1, 9), 1)[::2])
        p2 = axs[1, tw].pcolormesh(change_bw, cmap=cmaps['g_cm'])

    axs[0, 0].set_ylabel('p[H] old')
    axs[1, 0].set_xlabel('p[H] new')

    fig.colorbar(p1, ax=axs[0, 4])
    axs[0, 4].set_title('proportion go responses')
    fig.colorbar(p2, ax=axs[1, 4])

    if save:
        fig.savefig(path_figs + "Fig5ResponseByUpdate.pdf", bbox_inches='tight')

    return changes_list_fw, changes_list_bw


def compare_ddm_to_data(data, m1, m2, m3, sizes=fig_sizes, cmaps=colormaps):
    """
    Creates a series of plots to illustrate the behaviour of the drift diffusion models.
    :param data: the original data
    :param m1:  the data generated by the first ddm
    :param m2:  the data generated by the second ddm
    :param m3:  the data generated bu the third ddm
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return: Nothing
    """
    fig, axs = plt.subplots(4, 4, figsize=(sizes['width'] * 4, sizes['height'] * 4), sharex='row', sharey='row')

    VSS2021_fig1, axs_predictionW1 = plt.subplots(1, 3, figsize=(sizes['width'] * 3 * 0.8, sizes['height'] * 0.8),
                                                  sharex='all', sharey='all')
    VSS2021_fig2, axs_predictionW2 = plt.subplots(1, 3, figsize=(sizes['width'] * 3 * 0.8, sizes['height'] * 0.8),
                                                  sharex='all', sharey='all')

    # define the bins for count plots
    bins = np.linspace(0, 1, 20)

    # define response types
    responses = ['CORRECT REJECTION', 'FALSE ALARM', 'MISS', 'HIT']

    # define probability windows
    n_pwins = 10
    window_prob = np.linspace(min(data.sampleAccprobHit_01), max(data.sampleAccprobHit_01), n_pwins)

    for m_id, m_df in enumerate([m1.loc[data.index, :], m2.loc[data.index, :], m3.loc[data.index, :], data]):

        # first plot: distribution correct & incorrect reaction times
        # filter correct and incorrect reaction times
        RTs_c = m_df[m_df.answer == 1].rea_time
        RTs_i = m_df[m_df.answer == 0].rea_time

        axs[0, m_id].hist([RTs_c[~np.isnan(RTs_c)], RTs_i[~np.isnan(RTs_i)]], bins=bins, stacked=True,
                         color=cmaps['p_cm']([0.8, 0.2]), label=['correct', 'false'])  # bins = bins
        #

        # second plot: mean hit probabilities before each response
        # initialize data frames
        value_respcat_fw = pd.DataFrame()

        # time information relative to go response
        for timewin in range(1, 7):

            time_col = 'sampleAccprobHit_0{}'.format(timewin)

            # loop through response categories
            for respcat in responses:
                # get mean probability for category and time value
                value_respcat_fw.loc[respcat, timewin] = (np.mean(
                    m_df.loc[m_df.response_cat == respcat, time_col]) / 2) + 0.5

        # plot
        for row, c in zip(responses, cmaps['c_cm'](np.linspace(0, 1, len(responses)))):
            axs[1, m_id].plot(range(1, 7), value_respcat_fw.loc[row, :], label=row.lower(), color=c)

        # third plot: reproduce the decision after evidence pattern
        # translate df into long data frame

        # in order to match the trials later, we want an ID
        m_df['ids'] = m_df.index
        # melt the data frame into long format
        long_m_df = m_df.melt(id_vars=['goResp', 'hitGoal', 'ids'],
                              value_vars=['sampleAccprobHit_01', 'sampleAccprobHit_02', 'sampleAccprobHit_03',
                                          'sampleAccprobHit_04', 'sampleAccprobHit_05', 'sampleAccprobHit_06'],
                              var_name='sampleID', value_name='sampleAccprobHit')

        for tWin in [1, 2]:
            # define a panda data frame to store the values:
            # intitialize the data frames
            change_fw = pd.DataFrame()

            # filter data from this tw
            ctw_fw = long_m_df[long_m_df.sampleID == 'sampleAccprobHit_0{}'.format(tWin)]

            # filter data from the next tw
            ntw_fw = long_m_df[long_m_df.sampleID == 'sampleAccprobHit_0{}'.format(tWin + 1)]

            # go through all probabilites in tw 1
            for p_start in range(0, n_pwins - 1):

                pst_low = window_prob[p_start]
                pst_up = window_prob[p_start + 1]

                # get all IDs that fullfill the trial
                start_IDs_fw = \
                ctw_fw.iloc[np.where((ctw_fw.sampleAccprobHit >= pst_low) & (ctw_fw.sampleAccprobHit < pst_up))[0]][
                    'ids']

                # go through all probabilities in the next time window
                for p_end in range(0, n_pwins - 1):
                    pend_low = window_prob[p_end]
                    pend_up = window_prob[p_end + 1]

                    # get all ids that fullfill the end probability requirement
                    end_IDs_fw = ntw_fw.iloc[
                        np.where((ntw_fw.sampleAccprobHit >= pend_low) & (ntw_fw.sampleAccprobHit < pend_up))[0]]['ids']

                    # get the intercept between the two lists
                    ID_list_fw = np.intersect1d(start_IDs_fw, end_IDs_fw)

                    # filter the dataframe
                    df_fw = long_m_df.loc[np.where(long_m_df.ids.isin(ID_list_fw))]

                    change_fw.loc[window_prob[p_start], window_prob[p_end]] = np.mean(df_fw.goResp)

            # plot
            axs[1 + tWin, m_id].pcolormesh(change_fw, cmap=cmaps['g_cm'])
            if m_id <= 2:
                if tWin == 1:
                    axs_predictionW1[m_id].pcolormesh(change_fw, cmap=cmaps['g_cm'])
                else:
                    axs_predictionW2[m_id].pcolormesh(change_fw, cmap=cmaps['g_cm'])

    # Use absolute value for y-ticks
    # change the axis ticks
    ticks = axs[0, 0].get_yticks()
    axs[0, 0].set_yticklabels([np.round(abs(tick / len(m_df)), 2) for tick in ticks])
    axs[0, 3].legend()
    axs[0, 0].set_ylabel('response proportion')
    axs[0, 0].set_xlabel('response time in s')

    axs[1, 3].legend()
    axs[1, 0].set_ylabel('hit probability')
    ticks = axs[1, 0].get_xticks()
    axs[1, 0].set_xticklabels([tick / 10 for tick in ticks]);
    axs[1, 0].set_yticklabels(np.linspace(0, 1, 5));

    axs[1, 0].set_xlabel('time relative to go signal')

    axs[2, 1].set_yticklabels(np.round(change_fw.columns, 2)[::2])
    axs[2, 1].set_xticklabels(np.round(change_fw.index, 2)[[0, 2, 5, 7]]);
    axs[2, 0].set_ylabel('hit probability at t')
    axs[2, 0].set_xlabel('hit probability at t+1')

    # set the axis for individual figures
    # print(window_prob)

    axs_predictionW1[0].set_yticks(np.arange(0, 10, 2))
    axs_predictionW1[0].set_yticklabels(np.round(window_prob, 1)[::2])

    axs_predictionW1[0].set_xticks(np.arange(0, 10, 2))
    axs_predictionW1[0].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)
    axs_predictionW1[1].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)
    axs_predictionW1[2].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)

    axs_predictionW2[0].set_yticks(np.arange(0, 10, 2))
    axs_predictionW2[0].set_yticklabels(np.round(window_prob, 1)[::2])

    axs_predictionW2[0].set_xticks(np.arange(0, 10, 2))
    axs_predictionW2[0].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)
    axs_predictionW2[1].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)
    axs_predictionW2[2].set_xticklabels(np.round(window_prob, 1)[::2], rotation=45)

    axs_predictionW1[0].set_ylabel('p[H] sample 1')
    axs_predictionW1[0].set_xlabel('p[H] sample +1')

    axs_predictionW2[0].set_ylabel('p[H] sample 2')
    axs_predictionW2[0].set_xlabel('p[H] sample +1')

    VSS2021_fig1.savefig(path_figs + 'ddmPredicitions_win1.pdf', bbox_inches='tight')
    VSS2021_fig2.savefig(path_figs + 'ddmPredicitions_win2.pdf', bbox_inches='tight')

    plt.tight_layout()

    return None


def make_miscellaneous_figures(data, sizes=fig_sizes, cmaps=colormaps):
    """
    Generate and save some random figures that will be needed for the final plots
    :param data: the dataframe
    :param cmaps: a dictionary of colormaps
    :param sizes: a dictionary of sizes
    :return:
    """
    # colorbar for time
    fig, axs = plt.subplots(figsize=(3 * sizes['width'], 0.0))
    cb1 = mpl.colorbar.ColorbarBase(axs, cmap=cmaps['t_cm'], orientation='horizontal')

    fig.savefig(path_figs + 'SupFig1_timescale.pdf', bbox_inches='tight')

    # timeline with samples dots
    VVSS2021_timecourse, axs_timecourse = plt.subplots(1, 1, figsize=(3 * sizes['width'], sizes['height']))

    start = np.round(min(data.sampleTimeSecGo), 2)  # earliest sample, rounded
    end = np.round(max(data.sampleTimeSecGo), 2)  # latest sample, rounded
    n_sample = 6

    axs_timecourse.scatter(np.linspace(start, end, n_sample), [1] * n_sample, color='black', s=500)
    axs_timecourse.set_xlim(0, 1)
    axs_timecourse.set_ylabel('samples')
    axs_timecourse.set_xlabel('time since go signal [s]')

    VVSS2021_timecourse.savefig(path_figs + 'SupFig2_timecourse.pdf', bbox_inches='tight')
