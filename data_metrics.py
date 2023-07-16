from numpy import log10, var



def signal_to_noise(_noised, _original):
    signal = _original
    noise = _noised - signal

    signal_var = var(signal)
    noise_var = var(noise)
    return 10 * log10(signal_var/noise_var)