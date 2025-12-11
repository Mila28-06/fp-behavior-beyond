# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:26:28 2025

@author: conrad

to do: plot latency times per animal, 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from collections import defaultdict
import violinPlots
import scipy
import seaborn as sns
from boxoff import boxoff


def parse_latency_string(s):
    if pd.isna(s):
        return np.array([])  # Empty array if missing value
    
    # Force to string, remove Excel's forced-text apostrophes
    s = str(s).strip().lstrip("'")
    
    # Split on comma+space, then convert each to float
    values = []
    for part in s.split(','):
        part = part.strip()
        if part:  # Skip empty
            try:
                values.append(float(part))
            except ValueError:
                pass  # Or log a warning
    return np.array(values)


def expand_trials(df):
    expanded = []
    for _, row in df.iterrows():
        trials = [int(x) for x in row['trial class'].split('.')]
        latencies = (
            [float(x) for x in str(row['prey trial latency']).split(',')]
            if str(row['prey trial latency']).lower() != 'nan'
            else []
        )

        latency_iter = iter(latencies)
        for tnum, tclass in enumerate(trials, start=1):
            lat = next(latency_iter, np.nan) if tclass == 2 else np.nan
            expanded.append({
                'subject': row['ID'],
                'session_date': row['session_date'],
                'trial_num': tnum,
                'trial_class': tclass,
                'latency': lat
            })
    return pd.DataFrame(expanded)

data = defaultdict(lambda: {
    'aprch': 0,
    'preyTotal': 0,
    'IRaprch': 0,
    'IRtotal': 0,
    'p_latency': [],
    'IR_latency': []
})


filePath = 'W:\\Conrad\\Innate_approach\\Data_collection\\24.35.01\\'
savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\'
ntFilePth = 'W:\\Conrad\\Innate_approach\\Data_collection\\Neurotar\\'

r_log = pd.read_csv(f"{filePath}\\recordinglog.csv", sep=None, engine="python", encoding='cp1252')

r_log = r_log[r_log['Exp'] == 'nt']
r_log = r_log[pd.notna(r_log['trial class'])]
r_log = r_log[r_log['notes'] != 'no ttl alignment']



r_log = r_log.loc[:108]

animalIDs = r_log['ID'].unique()
dates = r_log['Date'].unique()

sex_dict = {
    "105647": "F",
    "105648": "F",
    "107818": "F",
    "107819": "F",
    "109436": "F",
    "111609": "M",
    "111610": "M",
    "112741": "M",
    "112742": "M",
    "115419": "M",
    "115424": "F",
    "116632": "F"
}

r_log['session_date'] = (
    r_log['Date']
    .astype(str)               # ensure string
    .str.strip('_')            # remove trailing underscore
    .str.replace('_', '-')     # convert to YYYY-MM-DD format
)
r_log['session_date'] = pd.to_datetime(r_log['session_date'], format='%Y-%m-%d')

# Recreate trial-level DataFrame if needed
trial_df = expand_trials(r_log)
trial_df['sex'] = trial_df['subject'].map(sex_dict)


latency_df = trial_df.dropna(subset=['latency'])

# Mean latency per trial number
latency_summary = latency_df.groupby('trial_num')['latency'].mean().reset_index()

# Compute each subject’s mean latency across sessions for each trial number
subj_avg = (
    latency_df.groupby(['subject', 'trial_num'])['latency']
    .mean()
    .reset_index()
)

plt.figure(figsize=(7,4))

# Plot faint lines for each subject’s average across sessions
sns.lineplot(
    data=subj_avg,
    x='trial_num',
    y='latency',
    hue='subject',
    estimator=None,   # plot raw subject means
    alpha=0.3,
    linewidth=1,
    legend=False
)

# Overlay grand mean ± SEM across subjects
sns.lineplot(
    data=subj_avg,
    x='trial_num',
    y='latency',
    errorbar='se',
    marker='o',
    color='black',
    linewidth=2
)

plt.title('Approach Latency vs Trial Number')
plt.xlabel('Trial Number within Session')
plt.ylabel('Mean Latency (s) ± SEM')
boxoff()
plt.tight_layout()
plt.show()


# Sort sessions per subject and assign session number
trial_df = trial_df.sort_values(['subject', 'session_date'])
trial_df['session_num'] = (
    trial_df.groupby('subject')['session_date']
    .transform(lambda x: pd.factorize(x, sort=True)[0] + 1)
)

p2_df = (
    trial_df.groupby(['subject', 'session_num'])
    .apply(lambda x: (x['trial_class'] == 2).mean())
    .reset_index(name='p_type2')
)

plt.figure(figsize=(7,4))
sns.lineplot(
    data=p2_df,
    x='session_num',
    y='p_type2',
    hue='subject',
    palette = 'muted',
    marker='o',
    alpha=0.7,
    legend = False
)
sns.lineplot(
    data=p2_df.groupby('session_num')['p_type2'].mean().reset_index(),
    x='session_num',
    y='p_type2',
    color='black',
    linewidth=2,
    label='Group mean'
)
plt.title('Probability of Approach across Session Number')
plt.xlabel('Session Number')
plt.ylabel('P(Approach)')
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'])
boxoff()
# plt.legend()
# plt.tight_layout()
plt.show()


for i in range(len(r_log)):
    row = r_log.iloc[i]

    if row['notes'] == 'no ttl alignment':
        continue

    idn = str(row['ID'])


    # prey
    p_trialClass = np.array([int(x) for x in str(row['trial class']).split('.')])
    ap_or_av = p_trialClass[(p_trialClass == 1) | (p_trialClass == 2)]
    if 2 in p_trialClass:
        p_latency = parse_latency_string(row['prey trial latency'])
        data[idn]['p_latency'].extend(p_latency[ap_or_av == 2])

    data[idn]['aprch']     += len(p_trialClass[p_trialClass == 2])
    data[idn]['preyTotal'] += len(p_trialClass)


    # IR
    IR_trialClass = np.array([int(x) for x in str(row['IR trial class']).split('.')])
    ap_or_av = IR_trialClass[(IR_trialClass == 1) | (IR_trialClass == 2)]
    if 2 in IR_trialClass:
        IR_latency = parse_latency_string(row['IR trial latency'])
        data[idn]['IR_latency'].extend(IR_latency[ap_or_av == 2])


    data[idn]['IRaprch'] += len(IR_trialClass[IR_trialClass == 2])
    data[idn]['IRtotal'] += len(IR_trialClass)


# # plot total values and calculate differences
# p_latency_agr = []
# IR_latency_agr = []
# p_prob = []
# IR_prob = []

# for i, idn in enumerate(data):
#     if idn == '105647' or idn == '109436':
#         continue
#     p_latency_agr.extend(data[idn]['p_latency'])
#     IR_latency_agr.extend(data[idn]['IR_latency'])
#     p_prob.append(len(data[idn]['p_latency'])/data[idn]['preyTotal'])
#     IR_prob.append(len(data[idn]['IR_latency'])/data[idn]['IRtotal'])

    
# # plot probability data
# violinPlots.violinPlots(p_prob, IR_prob, 'lightblue', 'orange', 'p', 
#                         'Probability of Approach','Probability', [0, 1],
#                         paired = True)

# scipy.stats.ttest_rel(p_prob, IR_prob, alternative = 'greater')

    
# # plot aggregate data
# # plot individual animal data
# fig, ax = plt.subplots(figsize=(10, 8))

# y = p_latency_agr
# x = np.zeros(len(y)) + 1

# if y:
#     # Violin plot for distribution
#     parts = ax.violinplot(
#                 y,
#                 positions=[1],
#                 showmeans=False,
#                 showextrema=False,
#                 showmedians=False
#             )

#     # Make the violin translucent
#     for pc in parts['bodies']:
#         pc.set_facecolor('lightblue')
#         pc.set_alpha(0.3)

#     # Scatter individual trials
#     ax.scatter(
#         x, y,
#         color='black',
#         alpha=0.5,
#         zorder=3,
#         label='Trials'
#     )

#     # Mean and SEM
#     mean_latency = np.mean(y)
#     sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))

#     ax.errorbar(
#         1, mean_latency,
#         yerr=sem_latency,
#         fmt='o',
#         color='red',
#         alpha = 0.5,
#         capsize=5,
#         markersize=8,
#         label='Mean ± SEM',
#         zorder=4
#     )


# y = IR_latency_agr
# x = np.zeros(len(y)) + 2

# # Violin plot for distribution
# if y:
#     parts = ax.violinplot(
#                 y,
#                 positions=[2],
#                 showmeans=False,
#                 showextrema=False,
#                 showmedians=False
#             )

#     # Make the violin translucent
#     for pc in parts['bodies']:
#         pc.set_facecolor('orange')
#         pc.set_alpha(0.3)

#     # Scatter individual trials
#     ax.scatter(
#         x, y,
#         color='black',
#         alpha=0.5,
#         zorder=3,
#         label='Trials'
#     )

#     # Mean and SEM
#     mean_latency = np.mean(y)
#     sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))

#     ax.errorbar(
#         2, mean_latency,
#         yerr=sem_latency,
#         fmt='o',
#         color='red',
#         alpha = 0.5,
#         capsize=5,
#         markersize=8,
#         label='Mean ± SEM',
#         zorder=4
#     )


# plt.xticks([1,2])
# plt.yticks([0, 5, 10, 15, 20])
# # ax.set_title("Aggregated Latency Data")
# # axs[i].set_xlabel("Prey vs IR laser")
# # ax.set_ylabel("Latency (s)")
# ax.set_xlim(0, 3)
# ax.set_ylim(0, 20)  
# # Hide the top and right spines
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)

# # Set tick parameters to remove right and top ticks
# plt.gca().tick_params(axis='x', which='both', direction='out', bottom=True, top=False)
# plt.gca().tick_params(axis='y', which='both', direction='out', left=True, right=False)

# scipy.stats.ttest_ind(p_latency_agr, IR_latency_agr)

#reformat data for stats
rows = []
trial_types_of_interest = {"p_latency", "IR_latency"}

for subj_id, trials in data.items():
    for trial_type, values in trials.items():
        if trial_type not in trial_types_of_interest:
            continue  
        
        if not isinstance(values, (list, tuple)):
            values = [values] if values is not None else []
        
        for value in values:
            rows.append({
                "subject_id": subj_id,
                "trial_type": trial_type,
                "response": value
            })

df = pd.DataFrame(rows)
df['sex'] = df['subject_id'].map(sex_dict)

df_filtered = df.groupby('subject_id').filter(lambda x: len(x) >= 3)

p_median = df_filtered.loc[df_filtered['trial_type'] == 'p_latency', 'response'].median()
IR_median = df_filtered.loc[df_filtered['trial_type'] == 'IR_latency', 'response'].median()

sns.lineplot(data=subj_avg, x='trial_num', y='latency',
             hue='subject', estimator=None, alpha=0.2, legend=False)

sns.lineplot(data=subj_avg, x='trial_num', y='latency',
             hue='sex', errorbar='se', linewidth=3)

model = smf.mixedlm(
    "response ~ trial_type",  # Fixed effect: trial_type
    data=df_filtered,
    groups=df_filtered["subject_id"]    # Random effect: subject
)
# model = smf.mixedlm("response ~ trial_type", df, groups=df["subject_id"], re_formula="1")


result = model.fit()
print(result.summary())

model = smf.mixedlm(
    "response ~ trial_type * sex",     # main effects + interaction
    data=df_filtered,
    groups=df_filtered["subject_id"]
)

result = model.fit()
print(result.summary())


# plot individual animal data
fig, axs = plt.subplots(3,3, figsize=(10, 8))
fig.suptitle('Latency data per animal')
axs = axs.flatten()

for i, idn in enumerate(data):
    if i >= len(axs):  # Avoid going past the subplot grid
       break
   
    y = data[idn]['p_latency']
    x = np.zeros(len(y)) + 1
    ax = axs[i]

    if y:
        # Violin plot for distribution
        parts = ax.violinplot(
                    y,
                    positions=[1],
                    showmeans=False,
                    showextrema=False,
                    showmedians=False
                )
    
        # Make the violin translucent
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.3)
    
        # Scatter individual trials
        ax.scatter(
            x, y,
            color='black',
            alpha=0.5,
            zorder=3,
            label='Trials'
        )
    
        # Mean and SEM
        mean_latency = np.mean(y)
        sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))
    
        ax.errorbar(
            1, mean_latency,
            yerr=sem_latency,
            fmt='o',
            color='red',
            alpha = 0.5,
            capsize=5,
            markersize=8,
            label='Mean ± SEM',
            zorder=4
        )

    
    y = data[idn]['IR_latency']
    x = np.zeros(len(y)) + 2
    
    # Violin plot for distribution
    if y:
        parts = ax.violinplot(
                    y,
                    positions=[2],
                    showmeans=False,
                    showextrema=False,
                    showmedians=False
                )
    
        # Make the violin translucent
        for pc in parts['bodies']:
            pc.set_facecolor('orange')
            pc.set_alpha(0.3)
    
        # Scatter individual trials
        ax.scatter(
            x, y,
            color='black',
            alpha=0.5,
            zorder=3,
            label='Trials'
        )
    
        # Mean and SEM
        mean_latency = np.mean(y)
        sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))
    
        ax.errorbar(
            2, mean_latency,
            yerr=sem_latency,
            fmt='o',
            color='red',
            alpha = 0.5,
            capsize=5,
            markersize=8,
            label='Mean ± SEM',
            zorder=4
        )

    axs[i].set_title(f"Animal {idn}")
    # axs[i].set_xlabel("Prey vs IR laser")
    axs[i].set_ylabel("Latency (s)")
    axs[i].set_xlim(0, 3)
    axs[i].set_ylim(0, 20)  

    
for ax in axs.flat:
    ax.label_outer()
    
    




