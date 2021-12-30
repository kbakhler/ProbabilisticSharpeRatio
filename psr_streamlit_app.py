import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PSR import *
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.sidebar.header('Track Record Stats:')
periods = st.sidebar.number_input(label='track record length',
                                  format='%i',
                                  value=36,
                                  min_value=1, step=1)
periodicity = st.sidebar.selectbox(label='periodicity',
                                   options=('daily', 'weekly', 'monthly'),
                                   index=2)
obsMean = st.sidebar.slider('mean return, observed',
                            value=0.0105,
                            min_value=-0.0500, max_value=0.0500, step=0.0005,
                            format=('%4f'),
                            help=('''non-annualized mean of returns
                                     distribution'''))
obsStd = st.sidebar.slider('st.dev. of returns, observed',
                           value=0.0150,
                           min_value=0.0000, max_value=0.2500, step=0.0005,
                           format=('%.4f'),
                           help=('''non-annualized standard deviation of returns
                                    distribution'''))
obsSkew = st.sidebar.slider('skewness of returns, observed',
                            value=-2.55,
                            min_value=-15.00, max_value=15.00, step=0.05,
                            help=('''Skewness of non-annualized returns
                                     distribution (e.g. skewness of Normal
                                     returns = 0)'''))
obsKurt = st.sidebar.slider('kurtosis of returns, observed',
                            value=4.15,
                            min_value=0.00, max_value=15.00, step=0.05,
                            help=('''Kurtosis of non-annualized returns
                                     distribution (e.g. kurtosis of Normal
                                     returns = 3)'''))
obsSR = obsMean/obsStd  # non-annuualized observed SR^
if periodicity == 'daily':
    freq = 250
elif periodicity == 'weekly':
    freq = 52
else:  # periodicity == 'monthly'
    freq = 12
ann_fact = freq**0.5
meanSR = obsSR*ann_fact

data_A = {"""A's Track Record (use left sidebar to customize)""":
          [periods, periodicity, obsMean, obsStd, obsSkew, obsKurt,
           np.round(meanSR, 4)]}
fundA_df = pd.DataFrame(data_A, index=['track record length',
                                       'annual periodicity',
                                       'mean return',
                                       'std of returns',
                                       'skewness of returns',
                                       'kurtosis of returns',
                                       'annualized Sharpe Ratio'])

st.write("""
# Probabilistic Sharpe Ratio (PSR)

Because a manager's "observed" Sharpe Ratio, $\widehat{SR}$, is only a single
point-estimate of his or her "true" Sharpe Ratio and subject to estimation
errors, a $PSR(0) > 0.95$ indicates that this manager's ${SR}$ is greater than
0 with a confidence level of 95%.  Similarly, a $PSR(1) > 0.9$ means the
manager's ${SR}$ is greater that 1.0 with 90% confidence.

${PSR}$ takes into account multiple statistical features of a manager's
track record - including its length, reporting frequency and deviations from
Normality (i.e. skewness, kurtosis).

${PSR}$ allows us to:
""")
st.write("""
1. Evaluate manager's track record vs a threshold Sharpe Ratio ($SR^*$)
2. Estimate minimum track record length ($minTRL$) needed for a threshold
Sharpe Ratio ($SR^*$)
3. Compare Sharpe Ratios of 2 different managers: ${PSR}[SR^*]_{A}$ vs
${PSR}[SR^*]_{B}$
""")
# 1. add expander for single-manager PSR evaluation
eval_expander = st.beta_expander("1. Evaluate Track Record vs Threshold SR",
                                 expanded=True)
with eval_expander:
    col_1a, col_1b = st.beta_columns([1, 2])
    with col_1a:
        st.write("""We can statistically compare a manager's track record
        (specified in sidebar on the left) to a threshold Sharpe Ratio, $SR^*$
        (specified below)...""")
        tSR_eval = st.number_input(label='threshold Sharpe Ratio',
                                   value=1.500000000000000000000000000000000,
                                   min_value=0.00,
                                   step=0.10000000000000000000000000000000000,
                                   help='annualized',
                                   format="%2f",
                                   key='tSR for eval')
        st.write("""...to see that the probability of $\widehat{SR} > SR^*$
        is: """)
        my_slot1a = st.empty()  # an empty slot for PSR[SR*]
    with col_1b:
        my_slot1b = st.empty()  # an empty slot for PSR chart
    col_1c, col_1d = st.beta_columns([1, 2])
    with col_1c:
        st.write("""Alternatively, we can estimate the lower confidence band for
        manager's Sharpe Ratio, given the track record (specified in
        sidebar on the left) and a confidence level (specified below)...""")
        conf_eval = st.number_input(label='confidence level',
                                    value=0.9500000000000000000000000000000000,
                                    min_value=0.00,
                                    max_value=1.00,
                                    step=0.01000000000000000000000000000000000,
                                    format="%2f",
                                    key='conf for eval')
        st.write("""...to see that the lower confidence band for $\widehat{SR}$
        is: """)
        my_slot1c = st.empty()  # an empty slot for PSR[SR*]
    with col_1d:
        my_slot1d = st.empty()  # an empty slot for PSR chart
# 2. add expander for min TrackRecordLength
mtrl_expander = st.beta_expander("2. Estimate Minimum Track Record Length",
                                 expanded=False)
with mtrl_expander:
    st.write("""Whereas the section above estimated the probability of
    $\widehat{SR} > SR^*$ using the actual track record length, this section
    answers the question:""")
    st.markdown("""> *“How long should a track record be in
    order to have statistical confidence that its Sharpe Ratio is above a given
    threshold?*”""")
    col_2a, col_2b = st.beta_columns([4, 6])
    with col_2a:
        st.write("""Considering the following track record:""")
        st.table(fundA_df)
    with col_2b:
        st.write("""...and specifying a threshold Sharpe Ratio $SR^*$:""")
        tSR_mtrl = st.number_input(label='threshold Sharpe Ratio',
                                   value=1.500000000000000000000000000000000,
                                   min_value=0.00,
                                   step=0.10000000000000000000000000000000000,
                                   help='annualized; must be << observed SR as \
                                   no length of track record will increase \
                                   observed SR above threshold',
                                   format="%.2f",
                                   key='tSR for mtrl')
        stats = [obsMean, obsStd, obsSkew, obsKurt]  # non-annualized stats
        sr_ref_mtrl = tSR_mtrl/ann_fact  # ref Sharpe ratio (non-annualized)
        # prob2 = conf_mtrl
        psr2a = PSR(stats, sr_ref_mtrl, periods, .5)
        # psr_mtrl = psr2.get_PSR(4)
        st.write("""...we know the probability of $\widehat{SR}>SR^*$
                 is""", str(np.round(psr2a.get_PSR(4)*100, 3))+"%",
                 """based on the current track record length of""",
                 str(periods), """periods (which is the same result as above).
                 But, if we also specify a desired confidence level:""")
        conf_mtrl = st.number_input(label='confidence level',
                                    value=0.9500000000000000000000000000000000,
                                    min_value=0.00,
                                    max_value=1.00,
                                    step=0.01000000000000000000000000000000000,
                                    format="%.2f",
                                    key='conf for mtrl')
        st.write("""...we'll see that we need returns for at least:""")
        psr2b = PSR(stats, sr_ref_mtrl, periods, conf_mtrl)
        st.success(np.round(psr2b.get_TRL(4), 2))
        st.write("""periods to feel confident that
        $\widehat{SR}$ (of""", str(np.round(meanSR, 2)), """) $>SR^*$
        (of""", str(np.round(tSR_mtrl, 2)), """) with
        statistical confidence of""", str(np.round(conf_mtrl*100, 3))+"%.")
        st.write("""*(note: minTRL is in the same periodicity as original track
        record entered on the left)*""")
# 3. add expander for Comparison of 2 Managers
compare_expander = st.beta_expander("3. Compare 2 Managers' Sharpe Ratios",
                                    expanded=False)
with compare_expander:
    st.write("""Since Sharpe Ratio estimates are subject to significant errors
    (due to non-Normality of returns and brief/irregular track records), it is
    prudent to consider ${PSR}$s and not just traditional ${SR}$s when
    comparing 2 managers.""")
    st.write("""Specifically, we are looking out for situations where
    ${PSR}[SR^*]_{A}<{PSR}[SR^*]_{B}$ despite
    $\widehat{SR}_{A}>\widehat{SR}_{B}$.""")
    col_3a, col_3b, col_3c = st.beta_columns([28, 45, 27])
    with col_3a:
        st.write("""We can statistically compare manager A's track record
        (specified in sidebar on the left) to manager B's
        (specified in column on the right) vs the same threshold Sharpe Ratio,
        $SR^*$ (specified below)...""")
        tSR_compare = st.number_input(label='threshold Sharpe Ratio',
                                      value=1.50000000000000000000000000000000,
                                      min_value=0.00,
                                      step=0.100000000000000000000000000000000,
                                      help='annualized',
                                      format="%2f",
                                      key='tSR for compare')
        st.write("""...to calculate and compare both ${PSR}[SR^*]_{A}$:""")
        my_slot3a = st.empty()  # an empty slot for PSR[SR*]_A
        st.write("""and ${PSR}[SR^*]_{B}$:""")
        my_slot3a2 = st.empty()  # an empty slot for PSR[SR*]_B
        my_slot3a3 = st.empty()  # an empty slot for
    with col_3b:
        my_slot3b = st.empty()  # an empty slot for table comparing 2 funds
    with col_3c:
        st.subheader('Track Record Stats: Fund B')
        periods_B = st.number_input(label='track record length',
                                    format='%i', value=24, min_value=1, step=1)
        periodicity_B = st.selectbox(label='periodicity',
                                     options=('daily', 'weekly', 'monthly'),
                                     index=2, key='periodicity_B')
        obsMean_B = st.slider('mean return, observed',
                              value=0.0125, min_value=-0.0500, max_value=0.050,
                              step=0.0005, format=('%4f'),
                              help=('''non-annualized mean of returns
                                    distribution'''))
        obsStd_B = st.slider('st.dev. of returns, observed',
                             value=0.0160, min_value=0.0000, max_value=0.2500,
                             step=0.0005, format=('%.4f'),
                             help=('''non-annualized standard deviation of
                                   returns distribution'''))
        obsSkew_B = st.slider('skewness of returns, observed',
                              value=-5.55, min_value=-15.00, max_value=15.00,
                              step=0.05,
                              help=('''Skewness of non-annualized returns
                                    distribution (e.g. skewness of Normal
                                    returns = 0)'''))
        obsKurt_B = st.slider('kurtosis of returns, observed',
                              value=8.15, min_value=0.00, max_value=15.00,
                              step=0.05,
                              help=('''Kurtosis of non-annualized returns
                                    distribution (e.g. kurtosis of Normal
                                    returns = 3)'''))
        obsSR_B = obsMean_B/obsStd_B  # non-annuualized observed SR^
        if periodicity_B == 'daily':
            freq_B = 250
        elif periodicity_B == 'weekly':
            freq_B = 52
        else:  # periodicity == 'monthly'
            freq_B = 12
        ann_fact_B = freq_B**0.5
        meanSR_B = obsSR_B*ann_fact_B
    my_slot3c = st.empty()  # an empty slot for PSR chart comparing 2 funds
# 4. add expander for References
ref_expander = st.beta_expander("References", expanded=False)
with ref_expander:
    st.write("""
    1. Bailey, David H. and López de Prado, Marcos, *The Sharpe Ratio Efficient
    Frontier*, 2012: available at [SSRN](https://ssrn.com/abstract=1821643).
    """)
    st.markdown("""> *Original paper introducing the Probabilistic Sharpe Ratio.
    Also includes python code for $PSR[SR]$ and $minTRL$ functions.*""")
    st.write("""
    2. Code for this dashboard and an accompanying Python notebook with
    mathematical formulas, additinal examples, etc is available at
    [Github](https://github.com/kbakhler/Probabilistic_Sharpe_Ratio).
    """)
    st.markdown("""> *Includes Python code for
    $\sigma_{\widehat{SR}}$ not included in the paper above as well as a
    slightly different treatment of ${PSR}$ as a Survival Function.*""")
    st.write("""
    3. Riondato, Matteo, *Sharpe Ratio: Estimation, Confidence Intervals, and
    Hypothesis Testing*, 2018: available at [Two Sigma]
    (https://www.twosigma.com/articles/sharpe-ratio-estimation-confidence-intervals-and-hypothesis-testing/).
    """)
    st.markdown("""> *While not directly linked to this dashboard or ${PSR}$, a
    good overview of various methods for Sharpe Ratio estimation,
    annualization, and estimation of its other statistical properties...listing
    various and often competing underlying assumptions.*""")


# start of custom function for stdev_of_obsSR
def stdev_of_obsSR(obsSR, obsSkew, obsKurt, n):
    """
    estimates st.dev. of the observed Sharpe Ratio (SR^)
    1) Inputs
    obsSR: observed non-annualized Sharpe Ratio
    obsSkew: observed skewness of returns (use 0 for Normal)
    obsKurt: observed kurtosis of returns (use 3 for Normal)
    n: number of observations
    2) Output
    sdSR: standard deviation of observed SR^ (non-annualized)
    """
    numerator = (1-(obsSkew*obsSR)+(((obsKurt-1)/4)*(obsSR**2)))
    denominator = (n-1)
    sdSR = (numerator/denominator)**.5
    return sdSR
# end of custom function for stdev_of_obsSR


significance_level = conf_eval  # for 95% one-sided conf
z_score_conf = norm.isf(significance_level)  # for one-sided conf
std = stdev_of_obsSR(obsSR, obsSkew, obsKurt, periods)*ann_fact
SR_lower_band_tsr = tSR_eval  # reference Sharpe ratio (annualized)
SR_lower_band_conf = meanSR+(z_score_conf*std)
x = np.linspace(meanSR-4*std, meanSR+4*std, 1000)
bellCurve = norm(meanSR, std)

# Plot A: create plot for PSR[SR*] with SR* specified by user
fig_tSR, ax_tSR = plt.subplots(figsize=(8, 6))
ax2_tSR = ax_tSR.twinx()  # twin object for 2nd y-axis on same plot
ax_tSR.plot(x, bellCurve.sf(x), color='orange', lw=2)
ax2_tSR.plot(x, bellCurve.pdf(x), color='white', lw=.1)
ax_tSR.set_xlabel("Sharpe Ratio", color='blue', fontsize=14)
ax_tSR.set_ylabel("PSR", color="blue", fontsize=14)
# fill-in under PDF: left-tail in pink, rest in green
x_left_tail = np.arange(meanSR-4*std, SR_lower_band_tsr, .01)
x_middle = np.arange(SR_lower_band_tsr, meanSR+4*std, .01)
ax2_tSR.fill_between(x_left_tail, bellCurve.pdf(x_left_tail), color='pink',
                     alpha=.7)
ax2_tSR.fill_between(x_middle, bellCurve.pdf(x_middle), color='springgreen',
                     alpha=.2)
# draw vertical and horizontal dashed lines
ax_tSR.plot([meanSR-4*std, SR_lower_band_tsr],
            [bellCurve.sf(SR_lower_band_tsr),
            bellCurve.sf(SR_lower_band_tsr)], color='tomato', ls=':')
ax_tSR.plot([SR_lower_band_tsr, SR_lower_band_tsr],
            [0, bellCurve.sf(SR_lower_band_tsr)], color='tomato', ls=':')
ax2_tSR.plot([meanSR, meanSR], [0, bellCurve.pdf(meanSR)],
             color='tomato', ls=':')
ax_tSR.plot([meanSR-4*std, meanSR+4*std], [0, 0], color='black', lw=1, ls='-')
# Annotation
props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.25}
ax_tSR.annotate('$PSR[{SR}^*]=$ %.3f' % (bellCurve.sf(SR_lower_band_tsr)),
                xy=(meanSR-4*std, bellCurve.sf(SR_lower_band_tsr)),
                xytext=(meanSR-3.9*std, bellCurve.sf(SR_lower_band_tsr)-0.1),
                va='top', fontsize=10,
                bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                'alpha': 1.0},
                arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
textstr_tSR = '\n'.join((
    r'$threshold$',
    r' '.join(('$Sharpe$', '$Ratio,$')),
    r'$annualized$',))
ax_tSR.annotate(textstr_tSR, xy=(SR_lower_band_tsr, 0.00),
                xytext=(SR_lower_band_tsr, 0.35), ha='center', va='center',
                fontsize=9)
ax_tSR.annotate('${SR}^*=$ %.2f' % (SR_lower_band_tsr),
                xy=(SR_lower_band_tsr, -0.05),
                xytext=(SR_lower_band_tsr, 0.27),
                ha='center', va='center', fontsize=11,
                bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                'alpha': 1.0},
                arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
textstr_obsSR = '\n'.join((
    r'$observed$',
    r' '.join(('$estimate$', '$of$')),
    r' '.join(('$Sharpe$', '$Ratio,$')),
    r'$annualized$',))
ax_tSR.annotate(textstr_obsSR, xy=(meanSR, 0.00),
                xytext=(meanSR, 0.16), ha='center', va='center', fontsize=9)
ax_tSR.annotate('$\widehat{SR}=$ %.2f' % (meanSR),
                xy=(meanSR, -0.05),
                xytext=(meanSR, 0.06),
                ha='center', va='center', fontsize=11,
                bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                'alpha': 1.0},
                arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, 0)})
# place an INPUTS text box in upper right in axes coords
textstr_inputs = '\n'.join((
    r'$\mathbf{INPUT:}$',
    r'# of obs$=%.0f$' % (periods, ),
    r'freq/yr$=%.0f$' % (freq, ),
    r'$\mathrm{mean}=%.4f$' % (obsMean, ),
    r'$\mathrm{s.d.}=%.4f$' % (obsStd, ),
    r'$\mathrm{skew}=%.2f$' % (obsSkew, ),
    r'$\mathrm{kurt}=%.2f$' % (obsKurt, ),
    r'$SR^*=%.3f$' % (tSR_eval, )))
ax_tSR.text(0.72, 0.97, textstr_inputs, transform=ax_tSR.transAxes,
            fontsize=12, va='top', bbox=props)
# place an OUTPUTS text box in center right in axes coords
textstr_outputs = '\n'.join((
    r'$\mathbf{OUTPUT:}$',
    # r'$\widehat{SR}=%.2f$' % (obsSR, ),
    # r'annualization',
    # r'    factor$=%.2f$' % (ann_fact, ),
    # r'$\widehat{SR}_a=%.2f$' % (meanSR, ),
    # r'$\widehat{\sigma}_{\widehat{SR}}=%.2f$' % (std, ),
    # r'${SR}^*=%.2f$' % (SR_lower_band_conf, ),
    r'$PSR[{SR}^*]=%.3f$' % (bellCurve.sf(SR_lower_band_tsr), ),))
ax_tSR.text(0.72, 0.6, textstr_outputs, transform=ax_tSR.transAxes,
            fontsize=12, va='top', bbox=props)
ax_tSR.xaxis.set_major_formatter(mtick.FormatStrFormatter('% 1.0f'))
ax_tSR.xaxis.set_major_locator(mtick.MultipleLocator(1.))
ax_tSR.xaxis.set_minor_locator(mtick.MultipleLocator(.25))
ax_tSR.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None,
                                 symbol='%', is_latex=False))
ax_tSR.yaxis.set_minor_locator(mtick.MultipleLocator(.05))
plt.yticks([])  # remove y-axis ticks/labels for ax2
ax_tSR.margins(x=0)
ax2_tSR.margins(x=0)
ax_tSR.margins(y=0.05)
ax2_tSR.margins(y=0.05)
ax_tSR.set_zorder(1)  # ensure SF plot on ax is visible above PDF plot on ax2
ax_tSR.patch.set_visible(False)  # prevents ax from hiding ax2

###
# Plot B: create plot for PSR[SR*] with Confidence_Level specified by user
fig_conf, ax_conf = plt.subplots(figsize=(8, 6))
ax2_conf = ax_conf.twinx()  # twin object for 2nd y-axis on same plot
ax_conf.plot(x, bellCurve.sf(x), color='orange', lw=2)
ax2_conf.plot(x, bellCurve.pdf(x), color='white', lw=.1)
ax_conf.set_xlabel("Sharpe Ratio", color='blue', fontsize=14)
ax_conf.set_ylabel("PSR", color="blue", fontsize=14)
# fill-in under PDF: left-tail in pink, rest in green
x_left_tail = np.arange(meanSR-4*std, SR_lower_band_conf, .01)
x_middle = np.arange(SR_lower_band_conf, meanSR+4*std, .01)
ax2_conf.fill_between(x_left_tail, bellCurve.pdf(x_left_tail), color='pink',
                      alpha=.7)
ax2_conf.fill_between(x_middle, bellCurve.pdf(x_middle), color='springgreen',
                      alpha=.2)
# draw vertical and horizontal dashed lines
ax_conf.plot([meanSR-4*std, SR_lower_band_conf],
             [bellCurve.sf(SR_lower_band_conf),
             bellCurve.sf(SR_lower_band_conf)], color='tomato', ls=':')
ax_conf.plot([SR_lower_band_conf, SR_lower_band_conf],
             [0, bellCurve.sf(SR_lower_band_conf)], color='tomato', ls=':')
ax2_conf.plot([meanSR, meanSR], [0, bellCurve.pdf(meanSR)],
              color='tomato', ls=':')
ax_conf.plot([meanSR-4*std, meanSR+4*std], [0, 0], color='black', lw=1, ls='-')
# Annotation
ax_conf.annotate('$PSR[{SR}^*]=$ %.3f' % (bellCurve.sf(SR_lower_band_conf)),
                 xy=(meanSR-4*std, bellCurve.sf(SR_lower_band_conf)),
                 xytext=(meanSR-3.9*std, bellCurve.sf(SR_lower_band_conf)-0.1),
                 va='top', fontsize=10,
                 bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                 'alpha': 1.0},
                 arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                 'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax_conf.annotate(textstr_tSR, xy=(SR_lower_band_conf, 0.00),
                 xytext=(SR_lower_band_conf, 0.35), ha='center', va='center',
                 fontsize=9)
ax_conf.annotate('${SR}^*=$ %.2f' % (SR_lower_band_conf),
                 xy=(SR_lower_band_conf, -0.05),
                 xytext=(SR_lower_band_conf, 0.27),
                 ha='center', va='center', fontsize=11,
                 bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                 'alpha': 1.0},
                 arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                 'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax_conf.annotate(textstr_obsSR, xy=(meanSR, 0.00),
                 xytext=(meanSR, 0.16), ha='center', va='center', fontsize=9)
ax_conf.annotate('$\widehat{SR}=$ %.2f' % (meanSR),
                 xy=(meanSR, -0.05),
                 xytext=(meanSR, 0.06),
                 ha='center', va='center', fontsize=11,
                 bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
                 'alpha': 1.0},
                 arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
                 'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, 0)})
# place an INPUTS text box in upper right in axes coords
textstr_inputs = '\n'.join((
    r'$\mathbf{INPUT:}$',
    r'# of obs$=%.0f$' % (periods, ),
    r'freq/yr$=%.0f$' % (freq, ),
    r'$\mathrm{mean}=%.4f$' % (obsMean, ),
    r'$\mathrm{s.d.}=%.4f$' % (obsStd, ),
    r'$\mathrm{skew}=%.2f$' % (obsSkew, ),
    r'$\mathrm{kurt}=%.2f$' % (obsKurt, ),
    r'$\mathrm{confidence}=%.3f$' % (significance_level, )))
ax_conf.text(0.72, 0.97, textstr_inputs, transform=ax_conf.transAxes,
             fontsize=12, va='top', bbox=props)
# place an OUTPUTS text box in center right in axes coords
textstr_outputs = '\n'.join((
    r'$\mathbf{OUTPUT:}$',
    # r'$\widehat{SR}=%.2f$' % (obsSR, ),
    # r'annualization',
    # r'    factor$=%.2f$' % (ann_fact, ),
    # r'$\widehat{SR}_a=%.2f$' % (meanSR, ),
    # r'$\widehat{\sigma}_{\widehat{SR}}=%.2f$' % (std, ),
    r'${SR}^*=%.3f$' % (SR_lower_band_conf, ),
    # r'$PSR[{SR}^*]=%.3f$' % (bellCurve.sf(SR_lower_band_conf), ),
    ))
ax_conf.text(0.72, 0.6, textstr_outputs, transform=ax_conf.transAxes,
             fontsize=12, va='top', bbox=props)
ax_conf.xaxis.set_major_formatter(mtick.FormatStrFormatter('% 1.0f'))
ax_conf.xaxis.set_major_locator(mtick.MultipleLocator(1.))
ax_conf.xaxis.set_minor_locator(mtick.MultipleLocator(.25))
ax_conf.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None,
                                  symbol='%', is_latex=False))
ax_conf.yaxis.set_minor_locator(mtick.MultipleLocator(.05))
plt.yticks([])  # remove y-axis ticks/labels for ax2
ax_conf.margins(x=0)
ax2_conf.margins(x=0)
ax_conf.margins(y=0.05)
ax2_conf.margins(y=0.05)
ax_conf.set_zorder(1)  # ensure SF plot on ax is visible above PDF plot on ax2
ax_conf.patch.set_visible(False)  # prevents ax from hiding ax2
#####

# Plot C: create plot for 2 PSR[SR*] for 2 Funds
######
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
meanSR1 = meanSR
std1 = std
meanSR2 = meanSR_B
std2 = stdev_of_obsSR(obsSR_B, obsSkew_B, obsKurt_B, periods_B)*ann_fact_B
SR_lower_band_compare_1 = tSR_compare
SR_lower_band_compare_2 = tSR_compare
x_compare = np.linspace(meanSR1-4*std1, meanSR1+4*std1, 1000)

bellCurve1 = norm(meanSR1, std1)
ax1.plot(x_compare, bellCurve1.sf(x_compare), color='orange', lw=2)
# ax1.set_xlabel("Sharpe Ratio",color='blue',fontsize=14)
ax1.set_ylabel("PSR: Fund A", color="blue", fontsize=14)
ax1b = ax1.twinx()
ax1b.plot(x_compare, bellCurve1.pdf(x_compare), color='white', lw=0.1)
bellCurve2 = norm(meanSR2, std2)
ax2.plot(x_compare, bellCurve2.sf(x_compare), color='orange', lw=2)
ax2.set_xlabel("Sharpe Ratio", color='blue', fontsize=14)
ax2.set_ylabel("PSR: Fund B", color="blue", fontsize=14)
ax2b = ax2.twinx()
ax2b.plot(x_compare, bellCurve2.pdf(x_compare), color='white', lw=0.1)
x_left_tail_compare_1 = np.arange(meanSR1-4*std1, SR_lower_band_compare_1,
                                  0.01)
x_middle_compare_1 = np.arange(SR_lower_band_compare_1, meanSR1+4*std1, 0.01)
ax1b.fill_between(x_left_tail_compare_1, bellCurve1.pdf(x_left_tail_compare_1),
                  color='pink', alpha=.7)
ax1b.fill_between(x_middle_compare_1, bellCurve1.pdf(x_middle_compare_1),
                  color='springgreen', alpha=.2)
x_left_tail_compare_2 = np.arange(max(meanSR2-4*std2, meanSR1-4*std1),
                                  SR_lower_band_compare_2, 0.01)
x_middle_compare_2 = np.arange(SR_lower_band_compare_2,
                               min(meanSR2+4*std2, meanSR1+4*std1), 0.01)
ax2b.fill_between(x_left_tail_compare_2, bellCurve2.pdf(x_left_tail_compare_2),
                  color='pink', alpha=.7)
ax2b.fill_between(x_middle_compare_2, bellCurve2.pdf(x_middle_compare_2),
                  color='springgreen', alpha=.2)
ax1.plot([meanSR1-4*std1, SR_lower_band_compare_1],
         [bellCurve1.sf(SR_lower_band_compare_1),
         bellCurve1.sf(SR_lower_band_compare_1)], color='tomato', ls=':')
ax1.plot([SR_lower_band_compare_1, SR_lower_band_compare_1],
         [0, bellCurve1.sf(SR_lower_band_compare_1)], color='tomato', ls=':')
ax1b.plot([meanSR1, meanSR1], [0, bellCurve1.pdf(meanSR1)],
          color='tomato', ls=':')
ax1.plot([meanSR1-4*std1, meanSR1+4*std1], [0, 0], color='black', lw=1, ls='-')
ax2.plot([meanSR1-4*std1, SR_lower_band_compare_2],
         [bellCurve2.sf(SR_lower_band_compare_2),
         bellCurve2.sf(SR_lower_band_compare_2)], color='tomato', ls=':')
ax2.plot([SR_lower_band_compare_2, SR_lower_band_compare_2],
         [0, bellCurve2.sf(SR_lower_band_compare_2)], color='tomato', ls=':')
ax2b.plot([meanSR2, meanSR2], [0, bellCurve2.pdf(meanSR2)],
          color='tomato', ls=':')
ax2.plot([meanSR1-4*std1, meanSR1+4*std1], [0, 0], color='black', lw=1, ls='-')
ax1.annotate(textstr_tSR, xy=(SR_lower_band_compare_1, 0.00),
             xytext=(SR_lower_band_compare_1, 0.63), ha='center', va='center',
             fontsize=8)
ax1.annotate(textstr_obsSR, xy=(meanSR1, 0.00),
             xytext=(meanSR1, 0.28), ha='center', va='center', fontsize=8)
ax2.annotate(textstr_tSR, xy=(SR_lower_band_compare_2, 0.00),
             xytext=(SR_lower_band_compare_2, 0.63), ha='center', va='center',
             fontsize=8)
ax2.annotate(textstr_obsSR, xy=(meanSR2, 0.00),
             xytext=(meanSR2, 0.28), ha='center', va='center', fontsize=8)
ax1.text(0.8, 0.8, "Fund A", transform=ax1.transAxes,
         fontsize=20, va='top', bbox=props)
ax2.text(0.8, 0.8, "Fund B", transform=ax2.transAxes,
         fontsize=20, va='top', bbox=props)
ax1.annotate('$PSR[{SR}^*]=$ %.3f' % (bellCurve1.sf(SR_lower_band_compare_1)),
             xy=(meanSR1-4*std1, bellCurve1.sf(SR_lower_band_compare_1)),
             xytext=(meanSR1-3.85*std1,
                     bellCurve1.sf(SR_lower_band_compare_1)-0.09),
             va='top', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax1.annotate('${SR}^*=$ %.2f' % (SR_lower_band_compare_1),
             xy=(SR_lower_band_compare_1, -0.05),
             xytext=(SR_lower_band_compare_1, 0.47),
             ha='center', va='center', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax1.annotate('$\widehat{SR}=$ %.2f' % (meanSR1),
             xy=(meanSR1, -0.05), xytext=(meanSR1, 0.08),
             ha='center', va='center', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, 0)})
ax2.annotate('$PSR[{SR}^*]=$ %.3f' % (bellCurve2.sf(SR_lower_band_compare_2)),
             xy=(meanSR1-4*std1, bellCurve2.sf(SR_lower_band_compare_2)),
             xytext=(meanSR1-3.85*std1,
                     bellCurve2.sf(SR_lower_band_compare_2)-0.09),
             va='top', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax2.annotate('${SR}^*=$ %.2f' % (SR_lower_band_compare_2),
             xy=(SR_lower_band_compare_2, -0.05),
             xytext=(SR_lower_band_compare_2, 0.47),
             ha='center', va='center', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, .1)})
ax2.annotate('$\widehat{SR}=$ %.2f' % (meanSR2),
             xy=(meanSR2, -0.05), xytext=(meanSR2, 0.08),
             ha='center', va='center', fontsize=9,
             bbox={'boxstyle': 'round', 'fc': 'dodgerblue', 'ec': 'none',
             'alpha': 1.0},
             arrowprops={'arrowstyle': 'simple', 'fc': 'dodgerblue',
             'ec': 'none', 'alpha': 1.0, 'relpos': (0.5, 0)})
for ax in (ax1, ax2):
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('% 1.0f'))
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1.))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(.25))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None,
                                 symbol='%', is_latex=False))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(.05))
    ax.set_zorder(1)
    ax.patch.set_visible(False)  # prevents ax from hiding ax-b
for ax in (ax1, ax1b, ax2, ax2b):
    ax.margins(x=0)
    ax.margins(y=0.05)
for ax in (ax1b, ax2b):
    ax.tick_params(right=False, labelright=False)
ax1.tick_params(bottom=True, labelbottom=True)
#################

# 1) Inputs (stats on excess returns) for evaluation of Manager A only
stats = [obsMean, obsStd, obsSkew, obsKurt]  # non-annualized stats
sr_ref = tSR_eval/ann_fact  # reference Sharpe ratio (non-annualized)
obs = periods
prob = conf_eval
# 2) Create class for evaluation of Manager A only
psr = PSR(stats, sr_ref, obs, prob)
# 3) Compute and report PSR[SR*] values for evaluation of Manager A only
psr_eval = psr.get_PSR(4)  # to use  in chart

# 1A) Inputs (stats on excess returns) for Manager A
sr_ref_A = tSR_compare/ann_fact_B  # reference Sharpe ratio (non-annualized)
# 2A) Create class for Manager A
psr_A = PSR(stats, sr_ref_A, obs, 0.5)
# 3A) Compute and report PSR[SR*] values for Manager A
psr_compare_A = psr_A.get_PSR(4)  # to use  in chart
# 1B) Inputs (stats on excess returns) for Manager A
stats_B = [obsMean_B, obsStd_B, obsSkew_B, obsKurt_B]  # non-annualized stats
sr_ref_B = tSR_compare/ann_fact_B  # reference Sharpe ratio (non-annualized)
obs_B = periods_B
prob_B = 0.5
# 2B) Create class for Manager B
psr_B = PSR(stats_B, sr_ref_B, obs_B, prob_B)
# 3B) Compute and report PSR[SR*] values for Manager B
psr_compare_B = psr_B.get_PSR(4)  # to use  in chart


def highlight_max(data, color="yellow"):
    """highlight the maximum in a Series or DataFrame"""
    attr = "background-color: {}".format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )


data_A2 = {"""A's Track Record (use left sidebar to customize)""":
           [periods, periodicity, obsMean, obsStd, obsSkew, obsKurt,
            np.round(meanSR, 4), np.round(psr_compare_A, 4)]}
data_B = {"""B's Track Record (use right column to customize)""":
          [periods_B, periodicity_B, obsMean_B, obsStd_B, obsSkew_B, obsKurt_B,
           np.round(meanSR_B, 4), np.round(psr_compare_B, 4)]}
data_AB = {**data_A2, **data_B}
funds_df = pd.DataFrame(data_AB, index=['track record length',
                                        'annual periodicity',
                                        'mean return',
                                        'std of returns',
                                        'skewness of returns',
                                        'kurtosis of returns',
                                        'annualized Sharpe Ratio',
                                        'PSR[SR*]'])
if psr_compare_A < psr_compare_B:
    if meanSR < meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}<{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}<\widehat{SR}_{B}$."""
    elif meanSR > meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}<{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}>\widehat{SR}_{B}$."""
    else:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}<{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}=\widehat{SR}_{B}$."""
elif psr_compare_A > psr_compare_B:
    if meanSR < meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}>{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}<\widehat{SR}_{B}$."""
    elif meanSR > meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}>{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}>\widehat{SR}_{B}$."""
    else:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}>{PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}=\widehat{SR}_{B}$."""
else:
    if meanSR < meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}={PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}<\widehat{SR}_{B}$."""
    elif meanSR > meanSR_B:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}={PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}>\widehat{SR}_{B}$."""
    else:
        psr_comparison_txt = """...to see that ${PSR}(""" + \
            str(np.round(tSR_compare, 3)) + \
            """)_{A}={PSR}(""" + str(np.round(tSR_compare, 3)) + \
            """)_{B}$ while $\widehat{SR}_{A}=\widehat{SR}_{B}$."""

my_slot1a.success(np.round(psr_eval, 3))
my_slot1b.write(fig_tSR)
my_slot1c.success(np.round(SR_lower_band_conf, 3))
my_slot1d.write(fig_conf)
my_slot3a.success(np.round(psr_compare_A, 3))
my_slot3a2.success(np.round(psr_compare_B, 3))
my_slot3a3.write(psr_comparison_txt)
my_slot3b.table(funds_df)
# my_slot3b.table(funds_df.style.apply(highlight_max, color="green", axis=1))
my_slot3c.write(fig)
