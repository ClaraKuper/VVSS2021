import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_summary_plot(df, cmap):
    """
    Creates a large plot summarizing behaviour in the experiment
    :param cmap: a dictionary holding all colormap information
    :param df: data frame to be visualized. Needs the columns "subject", "goResp", "reaTime", "hitGoal"
    :return: arrays needed to reproduce panels 2 and 3 (reaction times, performance in hit trial, performance in pass
    trials, observer array)
    """

    observers = np.unique(df.subject)

    # combined figure for visualization in notebook
    fig_description, axs_description = plt.subplots(1, 3, figsize=(20, 5))

    # set labels on combined figure
    # the first panel will be a graph of the number of go vs. no-go responses
    axs_description[0].set_title('Response frequencies')
    # the second panel will show the distribution of response times
    axs_description[1].set_title('Response times')
    # the third panel will show the proportion correct trial by condition
    axs_description[2].set_title('Performance')

    # plot panel 1
    # group by subject and go Response
    summary_a_bSG = df.groupby(['subject', 'goResp']).describe().answer
    # retrieve the count information from the summary
    go_count = summary_a_bSG.loc[((slice(observers[0], observers[-1])), [1]), :]['count'].values
    # flip one information
    nogo_count = -1 * summary_a_bSG.loc[((slice(observers[0], observers[-1])), [0]), :]['count'].values

    # plot everything
    axs_description[0].bar(observers, go_count, color=cmap['g_cm'](np.linspace(0.2, 0.8, 2)[1]), label='go')
    axs_description[0].bar(observers, nogo_count, color=cmap['g_cm'](np.linspace(0.2, 0.8, 2)[0]), label='no go')

    # Use absolute value for y-ticks
    ticks = axs_description[0].get_yticks()
    axs_description[0].set_yticklabels([np.round(int(abs(tick)) / (3 * 800), 1) for tick in ticks]);

    # add labels
    axs_description[0].set_ylabel('proportion of responses')
    axs_description[0].set_xlabel('observer')
    axs_description[0].legend(title='response', loc='upper right')

    # plot panel 2
    # collect reaction times in a list of lists
    rts = [df[df.subject == s].rea_time.values for s in np.unique(df.subject)]
    axs_description[1].hist(rts, stacked=True, color=cmap['s_cm'], label=observers)

    for s, c in zip(np.unique(df.subject), cmap['s_cm']):
        axs_description[1].axvline(x=np.mean(df[df.subject == s].rea_time), color=c)

    axs_description[1].legend(title='observer', loc='upper right')

    axs_description[1].set_xlabel('reaction time [s]')
    axs_description[1].set_ylabel('# responses')

    # plot panel 3
    summary_a_bSH = df.groupby(['subject', 'hitGoal']).describe().answer
    performance_hit = summary_a_bSH.loc[((slice(observers[0], observers[-1])), [1]), :]['mean'].values
    performance_pass = summary_a_bSH.loc[((slice(observers[0], observers[-1])), [0]), :]['mean'].values

    # for every subject and trial type
    axs_description[2].scatter(observers,
                               performance_hit,
                               label='hit', color=cmap['c_cm'](np.linspace(0.2, 0.8, 2))[1])
    axs_description[2].scatter(observers,
                               performance_pass,
                               label='pass', color=cmap['c_cm'](np.linspace(0.2, 0.8, 2))[0])
    axs_description[2].set_xlabel('observer')
    axs_description[2].set_ylabel('proportion correct')
    axs_description[2].legend(title='trial type', loc='lower right')

    plt.tight_layout()

    return rts, performance_hit, performance_pass, observers

