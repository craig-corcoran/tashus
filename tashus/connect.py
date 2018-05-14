import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np


dj.config['database.host'] = 'tutorial-db.datajoint.io'
dj.config['database.user'] = 'austinhilberg'
dj.config['database.password'] = 'Tdva82UA'

dj.conn()

schema = dj.schema('austinhilberg_tutorial', locals())


@schema
class Mouse(dj.Manual):
    definition = """
    # Mouse
    subject_name: varchar(128)  # mouse name
    """


@schema
class Session(dj.Manual):
    definition = """
    -> Mouse
    sample_number: int                    # sample used in session
    session_date: date                    # date of session
    ---
    n_stims: int                          # number of stimuli
    """


@schema
class Stimulation(dj.Imported):
    definition = """
    -> Session
    stim_number: int                # stimulus index within session
    ---
    fps: float                      # frames per second
    movie: longblob                 # stimulus movie
    n_frames: int                   # number of frames
    pixel_size: float               # in um/pixel
    stim_height: int                # in pixels
    stim_width: int                 # in pixels
    stimulus_onset: float           # from beginning of recording in seconds
    x_block_size: int               # pixels per horizontal block
    y_block_size: int               # pixels per vertical block
    n_neurons: int                  # number of neurons recorded
    """


@schema
class Spikes(dj.Imported):
    definition = """
    -> Stimulation
    neuron: int                 # neuron index within stimulus
    ---
    spikes: longblob            # recorded spikes
    """


@schema
class STADelays(dj.Lookup):
    definition = """
    stad_id: int           # int key for float params
    ---
    delay: float            # delay in seconds from stimulus to calculated STA
    """


@schema
class SpikeTriggeredAverage(dj.Computed):
    definition = """
    -> Spikes
    -> STADelays
    ---
    sta: longblob           # spike-triggered average for a given neuron and delay
    """

    def _make_tuples(self, key):
        if not (self & key).fetch(as_dict=True):
            stim = (Stimulation() & key).fetch1()
            spike_times = (Spikes() & key).fetch1('spikes')
            delay = (STADelays & key).fetch1('delay')
            spike_frames = np.round((spike_times - stim['stimulus_onset'] - delay) * stim['fps']).astype(int)
            spike_frames = spike_frames[spike_frames > 0 and spike_frames < stim['n_frames']]
            key['sta'] = np.mean(stim['movie'][:, :, spike_frames], axis=2)
            self.insert1(key, skip_duplicates=True)


mouse = Mouse()
session = Session()
stimulation = Stimulation()
spikes = Spikes()
sta_delays = STADelays()
sta = SpikeTriggeredAverage()


def init_setup(data_file='data.pkl'):

    records = np.load(data_file)

    for row in records:

        n_stimulations = len(row['stimulations'])
        session.insert1((
            row['subject_name'], row['sample_number'], row['session_date'], n_stimulations)
        )

        for i, stim in enumerate(row['stimulations']):
            n_spikes = len(stim['spikes'])

            stimulation.insert1((
                row['subject_name'], row['sample_number'], row['session_date'], i,
                stim['fps'], stim['movie'], stim['n_frames'], stim['pixel_size'],
                stim['stim_height'], stim['stim_width'], stim['stimulus_onset'],
                stim['x_block_size'], stim['y_block_size'], n_spikes
            ), skip_duplicates=True)

            for j, spike in enumerate(stim['spikes']):
                spikes.insert1((
                    row['subject_name'],
                    row['sample_number'],
                    row['session_date'],
                    i, j, spike
                ), skip_duplicates=True)


def plot_mouse(sample_dict):

    n_neurons = (stimulation & sample_dict).fetch1('n_neurons')

    for i in range(n_neurons):
        # for n in range(0, tmp_n):
        sample_dict['neuron'] = i
        for j in range(10):
            sample_dict['stad_id'] = j
            sta_sample = (sta & sample_dict).fetch1('sta').T

            plt.imshow(sta_sample, cmap='gray')

            plt.title('t - ' + str((sta_delays & sample_dict).fetch1('delay') * 1000) + ' ms')
            img_filename = '{session_date}_{subject_name}_{sample_number}_{stim_number}'.format(**sample_dict)
            img_filename += '_{0}_{1}.png'.format(i, j)
            plt.savefig(img_filename)


if __name__ == '__main__':

    # init_setup()

    plot_mouse(sample_dict={
        'subject_name': 'KO (chx10)',
        'sample_number': 4,
        'session_date': '2008-07-02',
        'stim_number': 1,
        'neuron': 0,
        'stad_id': 0}
    )
