from abc import ABC, abstractmethod
import et_literals as l
import sys
import et_consts as c
import numpy as np
import pandas as pd
from warnings import warn

sys.path.append(f"{c.project_dir}/Evaluation Framework")
from inputs_handler import InputsHandler
import ef_literals as efl


class SampleSelector(ABC):
    def __init__(self, questions, **kwargs):
        self.questions = questions
        self.agent_amount = kwargs[efl.AGENT_AMOUNT]
        self.existing_samples = kwargs[l.EXISTING_SAMPLES]
        self.agents_samples = kwargs[l.AGENTS_SAMPLES]

        exclude = [efl.AGENT_AMOUNT, efl.SEEDS, l.EXISTING_SAMPLES, l.AGENTS_SAMPLES]
        self.args = {k: v for k, v in kwargs.items() if k not in exclude}

        if 'seeds' in kwargs:
            self.seeds = kwargs['seeds']
        else:
            self.seeds_method = kwargs[efl.SEEDS_METHOD]
            self.seeds = (InputsHandler.generate_seeds(seeds_method=self.seeds_method, agent_amount=self.agent_amount))

    @abstractmethod
    def select(self):
        pass

    def samples_to_df(self, samples):
        samples_df = (pd.DataFrame.from_dict(samples, orient='index')
                      .reset_index(names=['agent'])
                      .melt('agent', value_name='questions_list')
                      .dropna(axis='index')
                      .sort_values(by=['agent', 'variable'])
                      .drop(columns='variable')
                      .reset_index(drop=True))
        samples_df['sample_code'] = samples_df['questions_list'].map(lambda x: "_".join(map(str, x)))

        # Join with existing samples
        texisting_samples = self.existing_samples.drop(columns='questions_list').set_index('sample_code')
        samples_df = samples_df.join(texisting_samples, how='left', on='sample_code')

        # Join with agents samples
        samples_df = pd.merge(left=samples_df, right=self.agents_samples, how='left',
                              left_on=['agent', 'sample_id'], right_on=['agent_id', 'sample_id'], indicator='Exist')
        samples_df = samples_df.drop(columns=['agent_id'])

        # Check if the sample exists
        samples_df['Exist'] = np.where(samples_df['Exist'] == 'both', True, False)
        # Generate samples' names
        samples_df['name'] = (samples_df['agent'].astype(str) + '_' +
                              samples_df['sample_id'].astype(str).replace('\.0', '', regex=True))
        return samples_df


class RandomSampleSelector(SampleSelector):
    """
    Select self.args['samples_amount'] random samples for each agent.
    Samples are generated randomly from the samples space, without considering the existing samples.
    """

    def select(self):
        lengths = np.random.randint(low=2,
                                    high=len(self.questions) + 1,
                                    size=self.agent_amount * self.args['samples_amount'])
        samples = {s: [] for s in self.seeds}
        for i, length in enumerate(lengths):
            sample = list(np.random.choice(self.questions, size=length, replace=False))
            aid = self.seeds[i // self.args['samples_amount']]
            samples[aid].append(sample)

        return samples


class RandomExistingSampleSelector(SampleSelector):
    """
    Select self.args['samples_amount'] random samples for each agent.
    Samples are selected randomly from the existing samples.
    If the agent doesn't have enough samples:
    1. If self.args['generate_missing'] is True, then the missing samples will be generated randomly.
    2. If self.args['generate_missing'] is False, then all the existing samples will be loaded.
    """

    def select(self):
        create_missing = self.args['create_missing']
        samples_p_agent = self.validate()
        agents_samples = self.agents_samples[self.agents_samples['agent_id'].isin(self.seeds)]

        enough_agents_samples = samples_p_agent[samples_p_agent['missing'] == 0]['agent_id']
        concat_dfs = []
        if (samples_p_agent['missing'] == 0).sum() > 0:
            s1 = (agents_samples[agents_samples['agent_id'].isin(enough_agents_samples)]
                  .groupby('agent_id')
                  .sample(self.args['samples_amount']))
            concat_dfs.append(s1)

        s2 = agents_samples[~agents_samples['agent_id'].isin(enough_agents_samples)]
        concat_dfs.append(s2)

        samples_dict = {}
        if len(concat_dfs) > 0:
            samples_df = pd.concat(concat_dfs)
            samples_df = samples_df.join(self.existing_samples.set_index('sample_id'), how='left', on='sample_id')
            samples_df = samples_df.groupby('agent_id')[['questions_list']].agg(samples=('questions_list', list))
            samples_dict.update(samples_df.to_dict()['samples'])

        if create_missing:
            for i, row in samples_p_agent[~samples_p_agent['agent_id'].isin(enough_agents_samples)].iterrows():
                agent_id = row['agent_id']
                s = RandomSampleSelector(questions=self.questions,
                                         seeds=[agent_id],
                                         agent_amount=1,
                                         samples_amount=row['missing'],
                                         existing_samples=self.existing_samples,
                                         agents_samples=self.agents_samples).select()
                if agent_id in samples_dict:
                    samples_dict[agent_id].extend(s[agent_id])
                else:
                    samples_dict.update(s)
        return samples_dict

    def validate(self):
        create_missing = self.args['create_missing']
        samples_p_agent = (self.agents_samples[self.agents_samples['agent_id'].isin(self.seeds)]
                           .groupby('agent_id')
                           .agg(amount=('sample_id', 'size'))
                           .reindex(self.seeds)
                           .fillna(0)
                           .astype(int)
                           .reset_index())
        samples_p_agent['missing'] = np.maximum(self.args['samples_amount'] - samples_p_agent['amount'], 0)
        if (samples_p_agent['missing'] > 0).any():
            if create_missing:
                last = "the missing samples will be generated randomly."
            else:
                last = "all the existing samples will be loaded."
            warning = (f"\nAgents with following ids doesn't exist or doesn't have enough samples!\n" +
                       f"\t{samples_p_agent[samples_p_agent['missing'] > 0]['agent_id'].to_list()}\n" +
                       f"For those, {last} The rest agents has enough samples:\n" +
                       f"\t{samples_p_agent[samples_p_agent['missing'] == 0]['agent_id'].to_list()}")
            warn(warning)
        return samples_p_agent


class WholeRandomSampleSelector(SampleSelector):
    pass


class StrategySampleSelector(SampleSelector):
    pass


sample_selectors = {l.SS.RANDOM: RandomSampleSelector,
                    l.SS.WHOLE_RANDOM: WholeRandomSampleSelector,
                    l.SS.STRATEGY: StrategySampleSelector,
                    l.SS.RANDOM_EXISTING: RandomExistingSampleSelector}

if __name__ == "__main__":
    # main
    args = {efl.AGENT_AMOUNT: 100,
            efl.SEEDS: efl.SD.ODD,
            l.SAMPLES_AMOUNT: 50,
            l.DEMO_STRATEGY: "dummy-4zm67j0v"}

    sample_selector = sample_selectors[l.SS.RANDOM](**args)
    sample_selector.select()
