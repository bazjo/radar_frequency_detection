import csv
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal

"""Run Configuration"""

empty_file = 'input_data/empty.csv'

"""Constants"""

CATCH_VALUES_REGEX = ',,adcBuf\[[0-9]{1,4}\],uint16_t,([0-9]{0,4})'

PULSE_SAMPLE_LENGTH = 1024
MAIN_BANG_LENGTH = 240  # Samples

FFT_SIZE = 8192

SAMPLE_RATE = 1e6  # Hz

DISTANCE_FREQUENCY_CONVERSION_FACTOR = (0.001 * 3e8) / (2 * 250e6)

MAX_DETECTION_DISTANCE = 8  # m


def read_pulse(input_file):
    pulse = []

    with open(input_file, newline='') as csv_file:
        pulse_reader = csv.reader(csv_file, delimiter=',', quotechar='\"')
        for row in pulse_reader:
            # print(','.join(row))
            regex_match = re.match(CATCH_VALUES_REGEX, ','.join(row))
            if regex_match is not None:
                # print(regex_match.group(1))
                pulse.append(int(regex_match.group(1)))

    # print(pulse)
    pulse = np.asarray(pulse)
    # print(pulse)
    return pulse


def frequency_calculation(input_file):
    """Calculate Process Variables"""

    fft_bins = FFT_SIZE / 2
    fft_resolution = SAMPLE_RATE / (2 * fft_bins)
    # Reference: https://electronics.stackexchange.com/questions/12407/what-is-the-relation-between-fft-length-and-frequency-resolution

    distance_resolution = DISTANCE_FREQUENCY_CONVERSION_FACTOR * fft_resolution
    max_detection_bin = int(MAX_DETECTION_DISTANCE / distance_resolution) + 1

    print('Input File Name:', input_file)

    print('FFT # of bins:', fft_bins)
    print('FFT bin resolution:', fft_resolution, 'Hz')

    print('Frequency to Distance:', DISTANCE_FREQUENCY_CONVERSION_FACTOR, 'm/Hz')
    print('Distance Resolution:', distance_resolution, 'm')

    """Read CSV Data into an Array"""

    pulse = read_pulse(input_file)
    empty = read_pulse(empty_file)

    """Process Data"""

    pulse = pulse - empty # subtract empty measurement

    pulse = pulse[MAIN_BANG_LENGTH:]  # remove main bang

    window = scipy.signal.windows.hann(PULSE_SAMPLE_LENGTH - MAIN_BANG_LENGTH, sym=False)  # calculate Hann window

    fft = scipy.fft.fft(pulse * window, n=FFT_SIZE) # perform FFT
    fft = np.absolute(fft) # get absolute values from complex
    fft = fft[:max_detection_bin] # cut fft to relevant part

    max_index = np.argmax(fft) # get the prevalent frequency within FFT

    detected_frequency = (max_index + 0.5) * fft_resolution
    detected_distance = detected_frequency * DISTANCE_FREQUENCY_CONVERSION_FACTOR

    snr = 10.0 * np.log10(fft[max_index] / np.average(fft))

    print('Frequency Peak detected at:', detected_frequency, 'Hz')
    print('Calculated Distance:', detected_distance, 'm')
    print('Signal to Noise Ratio:', snr, 'dB')

    plt.plot(fft)
    plt.plot(max_index, fft[max_index], "x")
    # plt.show()


if __name__ == "__main__":
    distances = [120, 250, 490, 600]

    for distance in distances:
        print('\n###\n')
        file_name = 'input_data/' + str(distance) + 'cm.csv'
        frequency_calculation(file_name)