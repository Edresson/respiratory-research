"""
This script creates usable data sets for audio recognition.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net

Note:
    Duplicate files are being ignored.
"""
import concurrent.futures
import os
import time
import glob
import shutil
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import pyloudnorm as pyln
import soundfile as sf
import subprocess
import sys

# TODO: move functions to independent modules

# 8 speeds between (0.8, 1.2); remove the speed with value 1
SPEEDS = np.delete(np.linspace(0.8, 1.2, 9), 4)

# 8 semitones between (-200, 200); remove the semitone with value 0
SEMITONES = np.delete(np.linspace(-200, 200, 9), 4)

NOISES = ['noises/ambiance.wav',
          'noises/crowd.wav',
          'noises/street.wav',
          'noises/driving.wav']

for noise in NOISES:
    if not os.path.isfile(noise):
        print(f'File {noise} missing. Noise processing might not work properly.')


def syscommand(command: str, debug: bool = False):
    """
    Runs a system command
    debug: bool
        If True, will print the command.
    return_out: bool
    """
    if debug:
        logm(command, cur_frame=currentframe())
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)
    stdout, stderr = process.communicate()
    if debug and stderr is not None:
        logm(stderr.decode(sys.stdout.encoding), mtype='E',
             cur_frame=currentframe())
    return stdout.decode(sys.stdout.encoding)


def get_audio_info(file_path, *args, verbose_level=0):
    """
    Get audio info.

    :param file_path: str
        Path of the audio file.
    :param args: *str
        List of key arguments. Currently supports 'duration' and 'rate'.
    :param verbose_level: int
        Verbosity level.

    :return: dict or str
        The returned value will be a dict with keys and the respective
        gathered information or only the information if you request a unique
        feature.
    """
    info = dict()
    value = None
    try:
        # To add new feature:
        # if 'feature' in args:
        #   run the command
        if 'duration' in args:
            value = syscommand('soxi -D {file}'.format(file=file_path),
                               debug=True if int(verbose_level) > 0
                               else False)
            info['duration'] = float(value)
        if 'rate' in args:
            value = syscommand('soxi -r {file}'.format(file=file_path),
                               debug=True if int(verbose_level) > 0
                               else False)
            info['rate'] = float(value)
    except ValueError as error:
        if str(verbose_level) == '2':
            print('[ERROR] Trying to get information of file {file}. '
                  '{error}.'.format(file=file_path, error=error))
        raise ValueError('Trying to get information of file {file}'.
                         format(file=file_path))
    # If just one value were requested, return a single value
    return info if len(info.keys()) > 1 else value


def files_to_process(files_list: list, output_dir: str) -> list:
    """
    Build a list of remaining files to process.

    This function checks the data sets integrity.

    :param files_list: list
        The original list of file names to process.
        Only names is considered by default (is_base_name=True).
    :param output_dir: str
        The path of the datasets (output directory).

    :return: list
        A list containing the remaining files to process (in the format of the
        provided list).
    """
    remaining_files = list()
    print('[INFO] checking directories in', output_dir)

    for file in files_list:
        if not os.path.isdir(output_dir + os.sep +
                             os.path.basename(file)[:-4]):
            remaining_files.append(file)

    print('There are {} files remaining to process'.
          format(len(remaining_files)))
    return remaining_files


def create_dataset(dataset_dir: str, file_list: list, num_workers: int = None,
                   pre_processing: callable = None, **kwargs):
    """
    Creates a dataset.

    :param dataset_dir: str
        Output directory (data set).
    :param file_list: list
        List of files to process (paths).
    :param num_workers: int
        The maximum number of processes that can be used to execute the given
        calls
    :param pre_processing: callable
        Pre process datasets before saving. Default to None, no pre processing
        will be performed.
    :param kwargs:
        Additional kwargs are passed on to the pre processing function.
    """
    if len(file_list) == 0:
        print('[WARN] no files to process the dataset {dataset}!'.
              format(dataset=dataset_dir))
    print('[INFO] creating data set {dataset}'.format(dataset=dataset_dir))
    os.makedirs(dataset_dir, exist_ok=True)

    if num_workers == 1:
        for file_path in file_list:
            # New feature (changed at 25/03) -> separate files by directory
            # pre_processing(file_path, dataset_dir, **kwargs)
            pre_processing(file_path, dataset_dir + os.sep + '_' +
                           os.path.basename(file_path)[:-4], **kwargs)
        return

    # Process data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as \
            executor:
        # New feature (changed at 25/03) -> separate files by directory
        # futures = [executor.submit(pre_processing, file_path,
        #                            dataset_dir, **kwargs)
        #            for file_path in file_list]
        futures = [executor.submit(pre_processing, file_path,
                                   dataset_dir + os.sep + '_' +
                                   os.path.basename(file_path)[:-4],
                                   **kwargs)
                   for file_path in file_list]

        kw = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass
        with open('script_create_dataset.txt', 'a') as log:
            log.write('\nExceptions for {function} call at '
                      '{time}'.format(function=pre_processing.__name__,
                                      time=time.time()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n{exception}\n\twhen processing {file}'
                              .format(exception=str(f.exception()),
                                      file=str(f)))


def pre_process(file_path: str, output_dir: str, name: str = None,
                min_length: float = 0, trim_interval: tuple = None,
                normalize_method: str = 'default', pitch_changing: float = None,
                noise_path: str = None, trim_silence_threshold: float = None,
                remix_channels: bool = False, speed_changing: float = None,
                robot: bool = False, rate: int = None, phone: bool = False,
                max_instances: int = None, low_pass_filter: float = None,
                ignore_length: bool = False, verbose_level=0, **kwargs):
    """
    Pre process a file. Use this function to handle raw datasets.

    This function generates temporary audio files. All files will be removed in
    the end of the process, except the original file provided and the final
    output file.

    :param file_path: str
        Path of the file to process.
    :param output_dir: str
        Path to save the processed file.
    :param name:
        Output name. Default to None (preserves the original name).
    :param min_length: float
        Minimum length of audio to consider. Skip otherwise.
    :param trim_interval: tuple, shape=(start, end)
        Trim audio. Default to None (no trim will be performed).
    :param normalize_method: str
        Normalization method.
        Normalize methods: Todo
    :param phone: bool
        Applies phone effect.
    :param robot: bool
        Applies robot effect.
    :param low_pass_filter: float
        Applies a low pass filter on.
    :param trim_silence_threshold: float.
        Threshold value to trim silence from audio. Default to None, no trim
        will be performed.
    :param rate: int
        Sets the rate of output file. Default to None (no conversion will be
        performed).
    :param remix_channels: bool
        Mix channels generating a mono audio. Default to True. It is1
        recommended to set this argument as True when the rate is converted.
    :param pitch_changing: int
        Integer representing the pitch changing effect applied to the audio.
        This changes the pitch of the audio, but not its speed.
    :param noise_path: str
        Path to an audio representing a noise for mixing with the original file.
    :param speed_changing: float
        Percentage representing the changing of the speed of the audio.
    :param max_instances: int
        Maximum number of processed instances. Default to none.
    :param ignore_length: bool
        If true, will pass --ignore_length to each audio, forcing the length
        checking. Can slow down the process.
    :param kwargs: dict
        Additional kwargs to pass on to the processing functions.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    # New feature (changed at 25/03) -> separate files by directory
    if max_instances and len(glob.glob(
            os.path.dirname(output_dir) + '*/**/*.wav', recursive=True)) >= \
            max_instances:
        if int(verbose_level) > 1:
            print('[INFO] maximum dataset size (maximum number of instances) '
                  'reached')
        return
    os.makedirs(output_dir, exist_ok=True)
    if int(verbose_level) > 1:
        print('[INFO] processing {file}'.format(file=file_path))

    # Create a set of temporary files
    temp_files = set()

    if min_length > 0:
        try:
            audio_length = float(get_audio_info(file_path, 'duration',
                                                verbose_level))
        except ValueError:
            return
        if audio_length is None:
            if str(verbose_level) == '2':
                print('[WARN] audio length of file {} is zero?'.
                      format(file_path))
            return

        expected_length = trim_interval[1] - trim_interval[0] if trim_interval \
                                                                 is not None else min_length
        if audio_length < expected_length and ignore_length:
            # Force the audio length checking (can be slow)
            if str(verbose_level) == '2':
                print('[WARN] forcing audio length checking of audio {file}'
                      ': Got {length} seconds'.format(file=file_path,
                                                      length=audio_length))
            tmp = output_dir + os.sep + 'tmp.wav'
            if os.path.isfile(tmp):
                if str(verbose_level) == '2':
                    print('[WARN] {tmp} already exists: {file} will be skipped '
                          'from processing'.format(tmp=tmp, file=file_path))
                return

            # This command call sox to rewrite the header
            cmd = 'sox -V{vlevel} --ignore-length {file} {tmp}'. \
                format(vlevel=verbose_level,
                       file=file_path,
                       tmp=tmp)
            if str(verbose_level) == '2':
                print(cmd)
            os.system(cmd)
            file_path = output_dir + os.sep + os.path.basename(file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)
            os.rename(tmp, file_path)
            temp_files.add(file_path)

            audio_length = float(get_audio_info(file_path, 'duration',
                                                verbose_level))
            if str(verbose_level) == '2':
                print('[INFO] fixed audio length of audio {file}'
                      ': Got {length} seconds'.format(file=file_path,
                                                      length=audio_length))
            if audio_length < expected_length and os.path.isfile(file_path):
                os.remove(file_path)
                if str(verbose_level) == '2':
                    print('[WARN] Length {length} of audio {file} is less than '
                          '{min} (ignoring)'.format(length=audio_length,
                                                    file=file_path,
                                                    min=min_length))
                return

        elif audio_length < expected_length:
            if str(verbose_level) == '2':
                print('[WARN] Length {length} of audio {file} is less than '
                      '{min} (ignoring)'.format(length=audio_length,
                                                file=file_path,
                                                min=min_length))
            return
    else:
        expected_length = None  # Variable not being used. Rare case.
    if name is None:
        # Todo: change file_path.split to os.path.basename(file_path)[:-4]...
        name = file_path.split(os.sep)[-1][:-4] + '_'
    # Process each transformation
    if trim_interval is not None and (speed_changing is None or
                                      pitch_changing is None or noise_path is None):
        # Process trim first if NOT performing data augmentation
        name += '_trim_' + str(trim_interval[0]) + '_' + str(
            trim_interval[1]) \
                + '_'
        file_path = trim(file_path, output_dir, name, trim_interval[0],
                         trim_interval[1] - trim_interval[0],
                         verbose_level)
        temp_files.add(file_path)
    if remix_channels:
        name += '_remix_'
        file_path = remix(file_path, output_dir, name, verbose_level)
        temp_files.add(file_path)
    if low_pass_filter is not None:
        name += '_lowpass' + str("%.2f" % low_pass_filter) + '_'
        file_path = lowpass(file_path, output_dir, low_pass_filter, name,
                            verbose_level)
        temp_files.add(file_path)
    if rate is not None:
        name += '_rate' + str(rate) + '_'
        file_path = convert_rate(rate, file_path, output_dir, name,
                                 verbose_level)
        temp_files.add(file_path)
    if speed_changing is not None:
        name += '_speed_' + str("%.2f" % speed_changing) + '_'
        file_path = speed(speed_changing, file_path, output_dir, name,
                          verbose_level)
        temp_files.add(file_path)
    if pitch_changing is not None:
        name += '_pitch_' + str(pitch_changing) + '_'
        file_path = pitch(pitch_changing, file_path, output_dir, name,
                          verbose_level)
        temp_files.add(file_path)
    if robot:
        name += '_robot_'
        file_path = robot_voice(file_path, output_dir, name, verbose_level)
    if phone:
        name += '_phone_'
        file_path = phone_voice(file_path, output_dir, name, verbose_level)
    if normalize_method != 'skip':
        name += '_norm_'
        target_n = kwargs['target_n'] if 'target_n' in kwargs else None
        file_path = norm(file_path, output_dir, name, target_n,
                         normalize_method, verbose_level)
        temp_files.add(file_path)
    if trim_silence_threshold is not None:
        name += '_trs_' + str(trim_silence_threshold) + '_'
        file_path = trim_silence_audio(trim_silence_threshold, file_path,
                                       output_dir, name, verbose_level)
        temp_files.add(file_path)
    if noise_path is not None:
        name += '_noise_' + \
                noise_path.replace('/', os.sep).split(os.sep)[-1]. \
                    split('.')[-2] + '_'
        file_path = add_noise(noise_path, file_path, output_dir, name,
                              verbose_level)
        temp_files.add(file_path)
    if trim_interval is not None and 'trim' not in name:
        # Process trim last if performing data augmentation
        name += '_trim_' + str(trim_interval[0]) + '_' + str(
            trim_interval[1]) \
                + '_'
        file_path = trim(file_path, output_dir, name, trim_interval[0],
                         trim_interval[1] - trim_interval[0],
                         verbose_level)
        temp_files.add(file_path)
    if len(temp_files) == 0 and int(verbose_level) > 1:
        print('[WARN] no pre processing was performed on file', file_path)
        return
    temp_files.remove(file_path)

    # Remove the temporary files
    for fp in temp_files:
        if os.path.isfile(fp):
            if int(verbose_level) == 2:
                print('[INFO] removing temporary file {}'.format(fp))
            os.remove(fp)


# Helper functions

def norm(file_path: str, output_dir: str, file_name: str, target_value: float,
         method='default', verbose_level: int = 0) -> str:
    """
    Normalizes an audio file.

    :param target_value: float
        Target value to apply the normalization or the parameter to configure
        the normalization. See method.
    :param method: str or callable
        Normalizes the audio file using the provided method.

        Options: 'default', 'peak', 'loudness'.

        Default will adjust all audio files with the same average amplitude.
        The choice "loudness" is a implementation of the ITU-R BS.1770-4. The
        choice "peak" is a implementation where all audio samples are
        normalized by an amount based on the peak value of each audio.
        "skip" will skip the normalization process.

        If callable, must return the path to the generated file.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Not being used.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    if method == 'default':
        if target_value is None:
            target_value = -20
        sound = AudioSegment.from_file(file_path, "wav")
        change_in_d_bfs = (target_value) - sound.dBFS
        sound = sound.apply_gain(change_in_d_bfs)
        sound.export(temp_file_path, format="wav")
    elif method == 'loudness':
        if target_value is None:
            target_value = -1
        data, rate = sf.read(file_path)
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness,
                                                            target_value)
        sf.write(temp_file_path, loudness_normalized_audio, rate)
    elif method == 'peak':
        if target_value is None:
            target_value = -1
        data, rate = sf.read(file_path)
        peak_normalized_audio = pyln.normalize.peak(data, target_value)
        sf.write(temp_file_path, peak_normalized_audio, rate)
    elif callable(method):
        return method(file_path, out_dir, file_name, target_value)
    else:
        raise ValueError('Invalid normalization method')
    return temp_file_path


def trim_silence_audio(trim_threshold: float, file_path, output_dir, file_name,
                       verbose_level: int = 0) -> str:
    """
    Removes silence.

    :param trim_threshold: float
        Below-periods duration threshold.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    #  -l 1.0 0.1 1% -1 2.0 1%
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    # todo: add silence threshold argument, in this case, it is set as 1.0
    cmd = 'sox -V{vlevel} {input} {output} ' \
          'silence {below_periods_duration} {above_periods} {duration} ' \
          '{duration_threshold}% {below_period} {ignore_period} ' \
          '{below_period_threshold}%' \
        .format(vlevel=str(verbose_level),
                input=file_path,
                output=temp_file_path,
                below_periods_duration='-l',
                # Remove silence or short silence
                # parameter. l means for not remove long periods of silence
                above_periods='1.0',  # Start removing silence from the
                # beginning
                duration='0.1',  # Minimum duration of silence to remove
                duration_threshold=trim_threshold,  # Trim silence
                # (anything less than 1% volume)
                below_period='-1',  # Remove silence until the end of the file
                ignore_period='2.0',  # Ignore small moments of silence
                below_period_threshold=trim_threshold
                )
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def remix(file_path, output_dir, file_name, verbose_level: int = 0) -> str:
    """
    Remix the audio into a single channel audio.

    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} remix 1'.format(vlevel=verbose_level,
                                                           input=file_path,
                                                           output=temp_file_path
                                                           )
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def convert_rate(rate, file_path, output_dir, file_name,
                 verbose_level: int = 0) -> str:
    """
    Remix the audio into a single channel audio.

    :param rate: int
        Integer representing the target rate. Example: 16000.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    # Convert the sample rate
    cmd = 'sox -V{vlevel} {input} -r {rate} {output}'. \
        format(vlevel=verbose_level,
               input=file_path,
               rate=rate,
               output=temp_file_path)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)

    return temp_file_path


def speed(param, file_path, output_dir, file_name,
          verbose_level: int = 0) -> str:
    """
    Speed up the provided audio.

    :param param: float
        Percentage of the speed.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} speed {param}'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path,
               param=param)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def pitch(param, file_path, output_dir, file_name,
          verbose_level: int = 0) -> str:
    """
    Change the audio pitch (but not tempo).

    :param param: int
        Integer representing the semitones to pitch the audio.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} pitch {param}'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path,
               param=param)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def add_noise(noise_path: str, file_path, output_dir, file_name,
              verbose_level: int = 0) -> str:
    """
    Adds noise to an audio file.

    :param noise_path: str
        Path to the noise file.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument are passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    # Check if the duration of the audio is less than the duration of the noise
    audio_info = get_audio_info(file_path, 'duration', 'rate', verbose_level)
    audio_length = float(audio_info['duration'])
    file_rate = float(audio_info['rate'])
    noise_info = get_audio_info(noise_path, 'duration', 'rate', verbose_level)
    noise_length = float(noise_info['duration'])
    noise_rate = float(noise_info['rate'])

    # Fix noise sample rate
    if noise_rate != file_rate:
        new_noise = noise_path[:-4] + '_remix_rate' + str(file_rate) + '.wav'
        if not os.path.isfile(new_noise):
            cmd = 'sox -V{vlevel} {input} -r {rate} {output} remix 1'. \
                format(vlevel=verbose_level,
                       input=noise_path,
                       rate=file_rate,
                       output=new_noise)
            if int(verbose_level) > 1:
                print('[INFO] The noise {noise} has sample rate of {noise_r}, '
                      'but the audio file has a rate of {file_rate}. '
                      'A new file will be generated at '
                      '{n_noise}'.format(noise=noise_path,
                                         noise_r=noise_rate,
                                         file_rate=file_rate,
                                         n_noise=new_noise))
            print(cmd)
            os.system(cmd)
        noise_path = new_noise

    temp_file_path = output_dir + os.sep + file_name + '.wav'
    if audio_length > noise_length:
        # Process mix with noise repeat1
        cmd = 'sox -V{vlevel} {noise} -r {rate} -p repeat {repeat} |' \
              ' sox -V{vlevel} - -m {input} {output}'. \
            format(vlevel=verbose_level,
                   noise=noise_path,
                   rate=file_rate,
                   repeat=int(audio_length / noise_length),
                   input=file_path,
                   output=temp_file_path)
    else:
        # Process mix
        cmd = 'sox -V{vlevel} -m {input} {noise} {output}'. \
            format(vlevel=verbose_level,
                   input=file_path,
                   noise=noise_path,
                   output=temp_file_path)

    if int(verbose_level) > 1:
        print(cmd)
    os.system(cmd)

    # Trim to the original time
    temp_file = trim(temp_file_path, output_dir, temp_file_path.
                     replace('/', os.sep).split(os.sep)[-1][:-4]
                     + '_temp', 0, float(get_audio_info(file_path, 'duration',
                                                        verbose_level)))

    if os.path.isfile(temp_file):
        os.remove(temp_file_path)
        os.rename(temp_file, temp_file_path)
    return temp_file_path


def phone_voice(file_path: str, output_dir: str, file_name: str,
                verbose_level: int = 0) -> str:
    """
    Applies a phone voice effect to an audio file.

    :param file_path: str
        Path to the audio to apply the effect.
    :param output_dir: str
        Output path to the output audio.
    :param file_name: str
        Output name.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} highpass 400 lowpass 3.4k'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def lowpass(file_path: str, output_dir: str, frequency: float, file_name: str,
            verbose_level: int = 0):
    """
    Applies a low pass filter to an audio file.

    :param file_path: str
        Path to the audio to apply the effect.
    :param output_dir: str
        Output path to the output audio.
    :param file_name: str
        Output name.
    :param frequency: float
        Frequency to apply the low pass filter.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} lowpass {value}'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path,
               value=frequency)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def robot_voice(file_path: str, output_dir: str, file_name: str,
                verbose_level: int = 0):
    """
    Applies a robot voice effect to an audio file.

    :param file_path: str
        Path to the audio to apply the effect.
    :param output_dir: str
        Output path to the output audio.
    :param file_name: str
        Output name.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    # cmd = 'sox -V{vlevel} {input} {output} overdrive 10 echo 0.8 0.8 5 0.7 ' \
    #       'echo 0.8 0.7 6 0.7 echo 0.8 0.7 10 0.7 echo 0.8 0.7 12 0.7    ' \
    #       'echo 0.8 0.88 12 0.7 echo 0.8 0.88 30 0.7 echo 0.6 0.6 60 0.7'. \
    cmd = 'sox -V{vlevel} {input} {output} overdrive 10 echo 0.8 0.8 5 0.7 ' \
          'echo 0.8 0.7 6 0.7 echo 0.8 0.7 10 0.7 echo 0.8 0.7 12 0.7    ' \
          'echo 0.8 0.88 12 0.7'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def trim(file_path: str, output_dir: str, file_name: str, position: float,
         duration: float, verbose_level: int = 0):
    """
    Trims an audio file using sox software.

    :param file_path: str
        Path to the audio to trim.
    :param output_dir: str
        Output path to the trimmed audio.
    :param file_name: str
        Output name.
    :param position:
        Start position to trim.
    :param duration:
        Duration in seconds
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} trim {position} {duration}'. \
        format(vlevel=verbose_level,
               input=file_path,
               output=temp_file_path,
               position=position,
               duration=duration)
    if int(verbose_level) > 1:
        print(cmd)
    os.system(cmd)
    return temp_file_path


def process_augmentation(dataset_dir: str, file_list: list,
                         num_workers: int = None,
                         pre_processing: callable = None,
                         **kwargs):
    print('[INFO] processing augmentation for dataset {dataset}'.
          format(dataset=dataset_dir))
    os.makedirs(dataset_dir, exist_ok=True)

    if num_workers == 1:
        for file_path in file_list:
            pre_processing(file_path,
                           dataset_dir + os.sep +
                           os.path.basename(os.path.dirname(file_path)),
                           **kwargs)
        return

    # Process data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as \
            executor:
        futures = [executor.submit(pre_processing, file_path,
                                   dataset_dir + os.sep +
                                   os.path.basename(os.path.dirname(file_path)),
                                   **kwargs)
                   for file_path in file_list]

        kw = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass


def augment_data(data_path: str, file_list: list, sliding_window: int = None,
                 trimming_window: int = None, seconds: float = 5,
                 noises: list = None, semitones: list = None,
                 speeds: list = None, robot: bool = False,
                 phone: bool = False, num_workers: int = None,
                 verbose_level: int = 0, low_pass_filter: float = None,
                 **kwargs):
    """
    Augments data by applying audio transformations.

    See pre_process for more information.

    :param data_path: str
        Path of the dataset. This will be the output path as well.
    :param file_list: list
        List of files to process.
    :param sliding_window: int
        Amount in seconds to slide and trim the audio. Use this for multiple
        small speeches.
    :param trimming_window: int
        Amount in seconds to slide and trim the audio. This process will run
        for each file and will generate fragments of each file depending on
        the length of it. Use this for large audio speeches.
    :param seconds: float
        Length of the audio in seconds to trim. Default to 5.
    :param noises: list
        List of noises paths to apply in each audio. Each noise will generate a
        new audio.
    :param semitones: list
        List of semitones to pitch in each audio. Each semitone will generate
        a new audio.
    :param speeds: list
        List of speeds to apply in each audio. Each speed will generate a new
        audio.
    :param low_pass_filter: float
        Applies a low pass filter of the provided frequency on audio files.
    :param robot: bool
        Add robot effect to audio files.
    :param phone: bool
        Add phone effect to audio files.
    :param num_workers: int
        Number of workers for multiprocessing .
    :param verbose_level: int
        Verbosity level.
    :param kwargs: dict
        Additional kwargs are passed on to the pre processing function.
    """
    if seconds is None and int(verbose_level) > 0:
        print('[WARN] seconds is not set (length to perform trimming '
              'operations)')
    if semitones is not None:
        print('[INFO] processing pitches')
        for i, p in enumerate(semitones):
            print('[INFO] processing pitches: {} of {}'.
                  format(i + 1, len(semitones)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 pitch_changing=p,
                                 **kwargs)
    if speeds is not None:
        print('[INFO] processing speeds')
        for i, s in enumerate(speeds):
            print('[INFO] processing speeds: {} of {}'.format(i + 1,
                                                              len(speeds)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 speed_changing=s,
                                 **kwargs)
    if noises is not None:
        print('[INFO] processing noises')
        for i, n in enumerate(noises):
            print('[INFO] processing noises: {} of {}'.format(i + 1,
                                                              len(noises)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 noise_path=n,
                                 min_length=seconds,
                                 **kwargs)
    if robot is not None:
        print('[INFO] processing robot voices')
        process_augmentation(dataset_dir=data_path,
                             file_list=file_list,
                             num_workers=num_workers,
                             pre_processing=pre_process,
                             verbose_level=verbose_level,
                             robot=True,
                             min_length=seconds,
                             **kwargs)
    if phone is not None:
        print('[INFO] processing robot voices')
        process_augmentation(dataset_dir=data_path,
                             file_list=file_list,
                             num_workers=num_workers,
                             pre_processing=pre_process,
                             verbose_level=verbose_level,
                             phone=True,
                             min_length=seconds,
                             **kwargs)
    if low_pass_filter is not None:
        print('[INFO] processing low pass filter')
        process_augmentation(dataset_dir=data_path,
                             file_list=file_list,
                             num_workers=num_workers,
                             pre_processing=pre_process,
                             verbose_level=verbose_level,
                             low_pass_filter=low_pass_filter,
                             min_length=seconds,
                             **kwargs)
    # Get files paths again and process sliding window or trimming to keep
    # a dataset with equal-length audio files
    # file_list = glob.glob(data_path + '/**/*.wav', recursive=True)
    file_list = []
    for dr in os.listdir(data_path):
        if dr[0] == '_':
            file_list += glob.glob(data_path + os.sep + dr + '*/**/*.wav',
                                   recursive=True)
    if sliding_window is not None:
        print('[INFO] processing sliding window')
        for i in range(0, 16, sliding_window):
            print('[INFO] operation {} of {}'.format(int(i / 2) + 1,
                                                     int(16 / 2 - 1) + 1))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 min_length=seconds,
                                 trim_interval=(0 + i, i + seconds),
                                 **kwargs)
    elif trimming_window is not None:
        print('[INFO] processing trimming window')
        if workers == 1:
            for audio in (tqdm(file_list) if int(verbose_level) < 2 else
            file_list):
                if int(verbose_level) > 0:
                    print('[INFO] processing trimming window of audio', audio)

                audio_length = float(get_audio_info(audio, 'duration',
                                                    verbose_level))
                # Changed: from range to np.arange
                for i in np.arange(0, audio_length - trimming_window,
                                   trimming_window):
                    if int(verbose_level) > 0:
                        print('[INFO] processing trimming window of {} '
                              '[trimming {} to {}] - [{}/{}]'.
                              format(audio, i, i + seconds, i,
                                     int(audio_length) - trimming_window))
                    pre_process(output_dir=data_path + os.sep +
                                           os.path.basename(os.path.
                                                            dirname(audio)),
                                file_path=audio,
                                verbose_level=verbose_level,
                                min_length=int(seconds),
                                trim_interval=(i, i + seconds),
                                **kwargs)
        else:
            for audio in (tqdm(file_list) if int(verbose_level) < 2 else
            file_list):
                # Process data in parallel
                process_trim_parallel(audio=audio,
                                      output_dir=data_path + os.sep +
                                                 os.path.basename(os.path.
                                                     dirname(
                                                     audio)),
                                      seconds=seconds,
                                      trimming_window=trimming_window,
                                      num_workers=num_workers,
                                      verbose_level=verbose_level,
                                      **kwargs)

    elif seconds is not None:
        print('[INFO] trimming audio files')
        create_dataset(dataset_dir=data_path,
                       file_list=file_list,
                       num_workers=num_workers,
                       pre_processing=pre_process,
                       verbose_level=verbose_level,
                       min_length=seconds,
                       trim_interval=(0, seconds),
                       **kwargs)

    if seconds:
        print('[INFO] removing old files from data augmentation')
        # The old files from data augmentation or post processing will be 
        # deleted. Note that if no trimming is performed the audio files will
        # have the same length as the original files, and will not be removed
        # by this code.
        for file in (tqdm(file_list) if int(verbose_level) < 2 else file_list):
            if os.path.isfile(file):
                if int(verbose_level) == 2:
                    print('[INFO] removing', file)
                os.remove(file)


def process_trim_parallel(audio, output_dir, seconds, trimming_window,
                          verbose_level, num_workers, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_workers)) \
            as executor:
        audio_length = float(get_audio_info(audio, 'duration', verbose_level))
        futures = [executor.submit(pre_process,
                                   output_dir=output_dir,
                                   file_path=audio,
                                   verbose_level=verbose_level,
                                   min_length=int(seconds),
                                   trim_interval=(i, i + seconds),
                                   **kwargs)
                   for i in np.arange(0, audio_length - trimming_window,
                                      trimming_window)]

        kw = {
            'total': len(futures),
            'unit': 'trims',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass


def make_json_file(directory):
    raise NotImplementedError


if __name__ == '__main__':
    # Todo: add command line options for data augmentation
    import json
    import argparse
    import importlib
    from collections import defaultdict
    from random import shuffle

    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format.')
    parser.add_argument('src',
                        help='Input Directory')
    parser.add_argument('out',
                        help='Output directory.')
    parser.add_argument('-n', '--normalization',
                        help='Normalize audio files. '
                             'Default will adjust all audio files with the '
                             'same average amplitude. The choice "loudness" '
                             'is a implementation of the ITU-R BS.1770-4. The '
                             'choice "peak" is a implementation where all '
                             'audio samples are normalized by an amount based '
                             'on the peak value of each audio. "skip" will '
                             'skip the normalization process.',
                        choices=['default', 'loudness', 'peak', 'skip'],
                        default='default')
    norm_args = parser.add_argument_group('Normalization options')
    norm_args.add_argument('--target_norm',
                           help='Desired target of the selected normalization '
                                'method. Available for "peak" and "loudness" '
                                'normalization.'
                                'The default for the "default" algorithms is '
                                '-20. The default for the "loudness" and '
                                '"peak" algorithms are -1.',
                           type=float)
    parser.add_argument('-s', '--seconds',
                        help='Length of audio files in seconds.',
                        type=float)
    parser.add_argument('-ts', '--trim_silence',
                        help='Volume threshold to trim silence from audio '
                             'files. Default to 0, no trim will be performed. '
                             'Recommended values: 1 - 5',
                        type=float,
                        default=None)
    parser.add_argument('-sr', '--samplerate',
                        help='Set the output rate of audio files.')
    parser.add_argument('-lp', '--lowpass',
                        help='Applies a low pass filter on audio files.',
                        type=float)
    parser.add_argument('-w', '--workers',
                        help='Define how many process to run in parallel.',
                        default=1,
                        type=int)
    parser.add_argument('-fmt', '--audioformat',
                        help='Set the audio format (extension) of the raw data.' 
                        ' Ex: wav. Default to all files in the provided path',
                        default='*')
    parser.add_argument('-c', '--check',
                        help='Check output directories ignoring files already '
                             'processed. Note: augmented data will not be '
                             'checked.',
                        action='store_true')
    parser.add_argument('-jc', '--just_check',
                        help='Check output directories. A csv log file will be '
                             'created containing the remaining files to '
                             'process. No processing will be performed. '
                             'Note: augmented data will not be checked.',
                        action='store_true')
    parser.add_argument('-l', '--limit',
                        help='Set a limit of files to process. Useful when the '
                             'data is unbalanced and the process is slow.'
                             'This limit is applied before the pre processing, '
                             'so it does not balances the data.',
                        type=int)
    parser.add_argument('-m', '--max',
                        help='Set a maximum number of processed files of each '
                             'class. Useful when the data is unbalanced and '
                             'the process is slow. This limit is applied '
                             'during the pre processing process. The number of '
                             'instances will be balanced if you provide a '
                             'number of instances that all class have. '
                             'It does not apply to augmented data.',
                        type=int)
    parser.add_argument('-lc', '--length_checking',
                        help='Will force length checking of each audio. '
                             'This will pass --ignore_length to sox. Can slow '
                             'down the process. Recommended if the processing '
                             'is skipping some audio files. ',
                        action='store_true')
    aug_args = parser.add_argument_group('Data augmentation options')
    aug_args.add_argument('-sw', '--sliding_window',
                          help='Sliding window: augments data by trimming '
                               'audio files in different positions. Must '
                               'provide the [seconds] argument',
                          action='store_true')
    aug_args.add_argument('-tw', '--trimming_window',
                          help='Trimming window:Trims audio files in different '
                               'positions. Use this if your dataset have large '
                               'speeches to process. This operation will trim '
                               'the original file in small speeches of '
                               'length [seconds]. Must provide the [seconds] '
                               'argument.',
                          action='store_true')
    aug_args.add_argument('-lpa', '--low_pass_augment',
                          help='Low pass filter: augments data by applying a '
                               'low pass filter on audio files.',
                          type=float)
    aug_args.add_argument('-sp', '--speed',
                          help='Speed: augments data by changing the speed of '
                               'the audio files.',
                          action='store_true')
    aug_args.add_argument('-ns', '--noise',
                          help='Noise: augments data by adding noise in the '
                               'audio files.',
                          action='store_true')
    aug_args.add_argument('-pt', '--pitch',
                          help='Pitch: augment data by changing the pitch of '
                               'audio files.',
                          action='store_true')
    aug_args.add_argument('--robot', help='Applies robot voice to audio files.',
                          action='store_true')
    aug_args.add_argument('--phone', help='Applies phone voice to audio files.',
                          action='store_true')
    parser.add_argument('-v', '--verbose',
                        help='Change verbosity level. Will affect all outputs.',
                        default=0)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.src
    out_dir = arguments.out
    duration = arguments.seconds
    output_rate = arguments.samplerate
    workers = arguments.workers
    verbose = arguments.verbose
    limit = arguments.limit
    trim_silence = arguments.trim_silence
    length_checking = arguments.length_checking
    normalization = arguments.normalization
    max_inst = arguments.max
    low_pass = arguments.lowpass
    low_pass_aug = arguments.low_pass_augment
    target_norm = arguments.target_norm
    audio_format = arguments.audioformat
    base = os.path.basename(data_dir)

    data_augmentation = arguments.pitch or arguments.speed or \
                        arguments.noise or arguments.sliding_window or \
                        arguments.trimming_window or arguments.low_pass_augment

    # Enable logging the remaining files with just check option.
    if arguments.just_check:
        os.makedirs('logs/scripts/', exist_ok=True)
        log_remaining = open('logs/scripts/dataset_remaining_files.csv', 'w')
        log_remaining.write('base' + ',' + 'original_base' + ',' + 'path')
    else:
        log_remaining = None
    
    if audio_format not in ['wav', 'mp3', 'sph', 'flac', 'aiff', 'ogg', 'aac',
                           'wma']:
        print('[WARN] unknown format of base')

    print('\n[INFO] getting a list of files of base "%s"' % base)
    # Get a list of all files (paths) to process
    all_files_path = glob.glob(data_dir + '/**/*.' + audio_format,
                               recursive=True)
    print('Total of raw files: %d' % len(all_files_path))

    # Check processed files if necessary, removing them
    if arguments.check or arguments.just_check:
        files_paths = files_to_process(all_files_path, out_dir)
    else:
        files_paths = None

    if log_remaining is not None and files_paths is not None:
        for f in files_paths:
            log_remaining.write('\n' + base + ',' + f)
    if files_paths is None:
        files_paths = all_files_path

    if limit is not None:
        print('[INFO] limiting the amount of files to process')
        shuffle(files_paths)
        files_paths = files_paths[:limit]

    print('TOTAL FILES TO BE PROCESSED: %d\n' % len(files_paths))

    if arguments.just_check:
        exit(0)

    print('[INFO] processing base "%s"' % base)

    if os.path.isdir(os.path.join(out_dir, base)):
        for dr in os.listdir(out_dir+ os.sep + base):
            if os.path.isdir(os.path.join(out_dir, base, dr)) and \
                    dr[0] == '_':
                if int(verbose) > 1:
                    print('[INFO] removing', os.path.join(out_dir, base, dr))
                shutil.rmtree(os.path.join(out_dir, base, dr))

    create_dataset(
        dataset_dir=out_dir + os.sep + base,
        file_list=files_paths,
        num_workers=workers,
        pre_processing=pre_process,
        verbose_level=verbose,
        min_length=duration if duration is not None else 0,
        rate=output_rate,
        normalize_method=normalization,
        target_n=target_norm,
        remix_channels=True,
        ignore_length=length_checking,
        trim_silence_threshold=trim_silence,
        max_instances=max_inst,
        low_pass_filter=low_pass if low_pass is not None else None,
        trim_interval=(0, duration)  # Disable trim if data augmentation is
        # enabled:
        if duration is not None and not data_augmentation else None)
    if max_inst:
        # When parallelism is enabled and the maximum size is set, there
        # might be excesses
        if not len(os.listdir(out_dir + os.sep + base)) < 1:
            excess = os.listdir(out_dir + os.sep + base)[max_inst:]
            shuffle(excess)
            if int(verbose) > 1:
                print('[INFO] deleting excess of base', base)
                for d in excess:
                    shutil.rmtree(out_dir + os.sep + base + os.sep + d)
        elif int(verbose) > 1:
            print('[WARN] empty folder', out_dir + os.sep + base)
    if data_augmentation:
        # The raw files will be removed, that is, all wav files in the
        # output folder, except the processed files.
        raw_files = []
        for dr in os.listdir(out_dir + os.sep + base):
            if dr[0] == '_':
                raw_files += glob.glob(out_dir + os.sep + base + os.sep + dr
                                        + '*/**/*.wav', recursive=True)
        print('[INFO] processing data augmentation')
        augment_data(out_dir + os.sep + base,
                        file_list=raw_files,
                        sliding_window=2 if arguments.sliding_window else None,
                        trimming_window=duration if arguments.trimming_window
                        else None,
                        seconds=duration,
                        noises=list(NOISES) if arguments.noise else None,
                        semitones=list(SEMITONES) if arguments.pitch else None,
                        speeds=list(SPEEDS) if arguments.speed else None,
                        robot=True if arguments.robot else None,
                        phone=True if arguments.phone else None,
                        num_workers=workers,
                        remix_channels=False,
                        low_pass_filter=low_pass_aug if low_pass_aug is not
                                                        None else None,
                        normalize_method='skip',
                        verbose_level=verbose)
        # Remove raw files
        # If the data augmentation is enabled, the length of each audio
        # file may vary because no trimming was performed before.
        # To maintain a standardized dataset it is necessary to remove the
        # old files.
        print('[INFO] removing old files')
        for fp in (tqdm(raw_files) if int(verbose) < 2 else raw_files):
            if os.path.isfile(fp):
                if int(verbose) == 2:
                    print('[INFO] removing', fp)
                os.remove(fp)

    # Update directories: remove the underscore of each folder. This means
    # that the process has done successfully.
    for dr in os.listdir(out_dir + os.sep + base):
        if len(os.listdir(out_dir + os.sep + base + os.sep + dr)) == 0:
            os.rmdir(out_dir + os.sep + base + os.sep + dr)
        elif dr[0] == '_':
            os.rename(out_dir + os.sep + base + os.sep +
                        dr, out_dir + os.sep + base + os.sep + dr[1:])
