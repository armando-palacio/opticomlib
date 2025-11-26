"""
.. rubric:: Functions
.. autosummary::

   search_inst
   connect_inst
   SYNC                  
   GET_EYE_v2
   save_h5
   load_h5            

.. rubric:: Classes
.. autosummary::

   PPG3204
   PED4002
   IDPhotonics         
   LeCroy_WavExp100H
"""
import numpy as np
import scipy.signal as sg
from scipy.stats import gaussian_kde

from .typing import binary_sequence, electrical_signal, eye
from typing import Literal

from .utils import (
    tic, toc, 
    str2array, 
    nearest,
    IntegerNumber,
    RealNumber,
    Iterable
)

import pyvisa as visa
import warnings
import re
import time
import h5py


def search_inst():
    """**Intruments search**
    
    Search for the available instruments in the system and print the IDs.
    """
    rm = visa.ResourceManager()
    print(rm.list_resources())

def connect_inst(addr_ID: str):
    """**Instrument connection**
    Connect to an instrument via VISA.
    Parameters
    ----------
    addr_ID : :obj:`str`
        VISA resource of the instrument (e.g. 'USB::0x0699::0x3130::9211219::INSTR').
    Returns
    -------
    inst : :obj:`visa.Resource`
        A connection (session) to the instrument.
    """
    inst = visa.ResourceManager().open_resource(addr_ID)
    inst.timeout = 10000 # timeout in milliseconds
    try:
        print(inst.query('*IDN?'))
    except:
        print('No identification received!!')
    return inst


def SYNC(signal_rx: electrical_signal | np.ndarray, 
         slots_tx: binary_sequence | np.ndarray, 
         sps: int = None):
    r"""**Signal Synchronizer**

    Synchronizes the received signal with the transmitted signal to determine the starting position in the received signal for further processing. 
    This is done by performing a correlation between the received signal and the transmitted signal and finding the maximum correlation position
    and shifting the received signal to that position (deleting the samples before the maximum correlation position).

    Parameters
    ----------
    signal_rx : :obj:`electrical_signal` | :obj:`np.ndarray`
        The received digital signal (from the oscilloscope or an ADC).
    slots_tx : :obj:`binary_sequence` | :obj:`np.ndarray`
        The transmitted slots sequence.
    sps : :obj:`int`, optional
        Number of samples per slot of the digitalized signal ``signal_rx``.

    Returns
    -------
    :obj:`tuple` [:obj:`electrical_signal`, :obj:`int`]
        A tuple containing the synchronized digital signal and the position in the ``signal_rx`` array from which synchronization was performed.

    Raises
    ------
    TypeError
        The ``sps`` must be an integer to perform synchronization.
    BufferError
        If the number of received slots have to be greater than the transmitted slots.
    ValueError
        If no correlation maximum is found.
    """
    
    tic()
    if isinstance(signal_rx, electrical_signal):
        sps = signal_rx.sps()
        signal_rx = signal_rx.signal
    elif isinstance(signal_rx, np.ndarray):
        if sps is None:
            raise ValueError('"sps" must be provided to perform synchronization.')
    else: 
        raise TypeError('The "signal_rx" must be of type `electrical_signal` or `np.ndarray`.')

    if isinstance(slots_tx, binary_sequence):
        slots_tx = slots_tx.data
    elif not isinstance(slots_tx, np.ndarray):
        raise TypeError('The "slots_tx" must be of type `binary_sequence` or `np.ndarray`.')

    signal_tx = np.kron(slots_tx, np.ones(sps))

    if len(signal_rx)<len(signal_tx): 
        raise BufferError('The length of the received vector must be greater than the transmitted vector!!')

    l = signal_tx.size
    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[l::-1], mode='valid') # Correlation of the transmitted signal with the received signal in a window of 2*l (sufficient to find a maximum)

    if np.max(corr) < 3*np.std(corr): 
        raise ValueError('No correlation maximum found!!') # false positive
    
    i = np.argmax(corr)

    signal_sync = electrical_signal(signal_rx[i:-(l-i)])
    signal_sync.execution_time = toc()
    return signal_sync, i


def GET_EYE_v2(
        sync_signal: electrical_signal | np.ndarray, 
        slots_tx: binary_sequence | np.ndarray, 
        nslots: int = 4096,
):
    r"""**Eye diagram parameters v2**

    Estimate the means and standard deviations of levels 0 and 1 in the ``sync_signal`` 
    by knowing the transmitted sequence ``slots_tx``. It separates the received signal levels
    corresponding to transmitted level 0 and 1 and estimates the means and standard deviations,
    different to ``devices.GET_EYE()`` that assume transmitted bits are not known. 

    Parameters
    ----------
    sync_signal : electrical_signal
        Synchronized digital signal in time with the transmitted signal.
    slots_tx : binary_sequence
        Transmitted bit sequence.
    nslots : int, default: 8192
        Number of slots to use for estimation.

    Returns
    -------
    dict
        A dictionary containing the following keys:

            - ``sps``: Samples per slot of the digital signal.
            - ``y``: Synchronized digital signal.
            - ``unos``: Received signal levels corresponding to transmitted level 1.
            - ``zeros``: Received signal levels corresponding to transmitted level 0.
            - ``t0``: Time instants for level 0.
            - ``t1``: Time instants for level 1.
            - ``i``: Position in the 'signal' vector from which synchronization was performed.
            - ``mu0``: Mean of level 0.
            - ``mu1``: Mean of level 1.
            - ``s0``: Standard deviation of level 0.
            - ``s1``: Standard deviation of level 1.
    """
    tic()

    #########################
    ## Preprocessing input ##
    #########################
    
    input = sync_signal
    
    if not isinstance(input, electrical_signal):
        input = electrical_signal(input)
    
    if not isinstance(slots_tx, binary_sequence):
        slots_tx = binary_sequence(slots_tx)

    eye_dict = {}

    eye_dict['sps'] = sps = input.sps()
    eye_dict['dt'] = dt = input.dt()

    # truncate
    n = input.len() % (2 * sps)  # we obtain the rest %(2*sps)
    if n: # if rest is not zero
        input = input[:-n] # ignore last 'n' samples

    nslots = min( int(input.len() // sps), nslots) # determine the minimum between slots of signal and 'nslots' parameter
    input = input[: nslots * sps] # truncate signal

    input = (input.signal + input.noise).real if input.noise is not None else input.signal.real # add noise to signal, if there is noise

    eye_dict["y"] = np.roll(input, -sps // 2 + 1)
    eye_dict['t'] = t = np.kron(np.ones(nslots // 2), np.linspace(-1, 1 - 1/sps, 2 * sps), )
    
    ###############
    ## Algorithm ##
    ###############

    ref = np.kron(slots_tx.data[:nslots], np.ones(sps))

    eye_dict['ones'] = ones = input[ref==1]
    eye_dict['zeros'] = zeros = input[ref==0]

    eye_dict['t0'] = t0 = np.kron(np.ones(zeros.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False))
    eye_dict['t1'] = t1 = np.kron(np.ones(ones.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False))

    eye_dict['i']=sps//2
    eye_dict["t_left"] = -0.5
    eye_dict["t_right"] = 0.5

    eye_dict["y_left"] = None
    eye_dict["y_right"] = None

    eye_dict["t_dist"] = t_dist = 1
    eye_dict["t_opt"] = t_opt = 0
    eye_dict["t_span0"] = t_span0 = t_opt - 0.05 * t_dist
    eye_dict["t_span1"] = t_span1 = t_opt + 0.05 * t_dist

    ones_ = ones[(t1>t_span0) & (t1<t_span1)]
    zeros_ = zeros[(t0>t_span0) & (t0<t_span1)]

    eye_dict['mu0'] = mu0 = np.mean(zeros_).real
    eye_dict['mu1'] = mu1 = np.mean(ones_).real

    eye_dict['s0'] = s0 = np.std(zeros_).real
    eye_dict['s1'] = s1 = np.std(ones_).real

    # compute umbral
    x = np.linspace(mu0, mu1, 500)
    pdf = gaussian_kde(zeros_.tolist() + ones_.tolist()).evaluate(x)
    eye_dict["threshold"] = x[np.argmin(pdf)]

    # We obtain the extinction ratio
    eye_dict["er"] = 10 * np.log10(mu1 / mu0) if mu0 > 0 else np.inf if mu0 == 0 else np.nan

    # We obtain the eye opening
    eye_dict["eye_h"] = mu1 - 3 * s1 - mu0 - 3 * s0
    
    eye_dict["execution_time"] = toc()
    return eye(**eye_dict)


def save_h5(filename, **datos):
    """
    Saves measurement data of signals in an HDF5 file.

    This function creates an HDF5 file that contains the common time vector,
    wavelengths, signal data matrix and optional metadata
    from the oscilloscope and experiment setup.

    Parameters
    ----------
    filename : str
        Base name of the file (without extension). '.h5' will be added.
    **datos : dict
        Name and value of the parameter to save. For example save_h5('name', time=t, wavelength=w)
    """
    with h5py.File(filename + '.h5', 'w') as f:
        for k,v in datos.items():
            if k != 'metadata':
                chunks = True if np.asarray(v).ndim>1 else None
                f.create_dataset(k, data=v, compression=None, chunks=chunks)

        metadata = datos.get('metadata', {})
        # metadata como atributos
        meta_grp = f.create_group('metadata')
        for k,v in metadata.items():
            meta_grp.attrs[k] = str(v)


def load_h5(filename):
    """
    Loads all datasets and metadata from an HDF5 file in a generic way.

    This function reads an HDF5 file and returns a dictionary with all datasets
    found (arrays loaded into memory) and metadata if they exist.

    Parameters
    ----------
    filename : str
        Base name of the file (without extension). '.h5' will be added.

    Returns
    -------
    data : dict
        Dictionary with dataset names as keys and ndarray values.
        If a 'metadata' group exists, includes a 'metadata' key with dict of attributes.
    """
    with h5py.File(filename + '.h5', 'r') as f:
        data = {}
        # Cargar todos los datasets de nivel superior
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data[key] = f[key][:]  # copia a memoria
            elif isinstance(f[key], h5py.Group) and key == 'metadata':
                # Cargar metadatos como dict
                metadata = {k: f[key].attrs[k].decode('utf-8') if isinstance(f[key].attrs[k], bytes) else f[key].attrs[k]
                            for k in f[key].attrs}
                data['metadata'] = metadata
    return data


class PPG3204():
    """**Tektronix Programmable Pattern Generator PPG3204**
    
    The `PPG3204 <https://download.tek.com/manual/PPG1600-PPG3000-PPG3200-Pattern-Generator-User-Manual-077109001.pdf>`_ 
    is a Programmable Pattern Generator. It is a 4-channel pattern generator with 32 Gb/s maximum data rate. 
    This class provides a set of methods to control the PPG3204.

    .. image:: _images/lab/PPG3204.png 
        :width: 80%
        :align: center

    The PPG3204 has the following features:
        
    .. rubric:: Attributes
    .. autosummary::

        ~PPG3204.inst
        CHANNELS
        PATT_LEN_MIN
        PATT_LEN_MAX
        AMPLITUDE_MIN
        AMPLITUDE_MAX
        OFFSET_MIN
        OFFSET_MAX
        FREQ_MIN
        FREQ_MAX
        PATT_TYPE
        PRBS_ORDERS
        MAX_MEMORY_LEN
        MAX_CHUNK_LEN
        MIN_SKEW
        MAX_SKEW
        
    .. rubric:: Methods
    .. autosummary::

        __init__
        __call__
        reset
        patt_len
        get_patt_len
        patt_type
        get_patt_type
        prbs
        get_prbs
        data
        get_data
        bits_shift
        get_bits_shift
        output
        get_output
        data_rate
        get_data_rate
        skew
        get_skew
        amplitude
        get_amplitude
        offset
        get_offset
        setup
        get_metadata
        print_setup
    """
    CHANNELS = 4 
    """Number of channels of the PPG3204, 4 channels."""
    PATT_LEN_MIN = 2
    """Pattern length minimum value, 2 bit."""
    PATT_LEN_MAX = 2**21
    """Pattern length maximum value, 2^21 = 2097152 (2M) bits."""
    AMPLITUDE_MIN = 0.3
    """Minimum amplitude of the output signal, 0.3 V."""
    AMPLITUDE_MAX = 2
    """Maximum amplitude of the output signal, 2 V."""
    OFFSET_MIN = -2
    """Minimum offset of the output signal, -2 V."""
    OFFSET_MAX = 3
    """Maximum offset of the output signal, 3 V."""
    FREQ_MIN = 1.5e9
    """Minimum frequency, 1.5 GHz."""
    FREQ_MAX = 32e9
    """Maximum frequency, 32 GHz."""
    PATT_TYPE = ['DATA', 'PRBS']
    """Mode of the pattern generator, ['DATA', 'PRBS']"""
    PRBS_ORDERS = [7,9,11,15,23,31]
    """The order of polynomial generator for PRBS_TYPE, [7,9,11,15,23,31]"""
    MAX_MEMORY_LEN = 2**21
    """Maximum length of the memory of the PPG3204, 2^21 = 2097152 (2M) for each channel."""
    MAX_CHUNK_LEN = 1024
    """Maximum length of the data to send in a single command, 1024 bits."""
    MIN_SKEW = -25e-12
    """Minimum skew, -25 ps"""
    MAX_SKEW = 25e-12
    """Maximum skew, 25 ps"""

    def __init__(self, addr_ID: str = None, reset: bool=True):
        """ Initialize the PPG3204.
        
        If ``addr_ID`` is not passed as argument, methods will print the commands 
        instead of sending them to the PPG. This is useful for debugging. 

        Parameters
        ----------
        addr_ID : :obj:`str`, optional
            VISA resource of the PPG (e.g. 'USB::0x0699::0x3130::9211219::INSTR'). Default is None.
        """
        if addr_ID: 
            self.inst = visa.ResourceManager().open_resource(addr_ID)
            """A connection (session) to the PPG instrument (if `addr_ID` is provided)."""
            self.inst.timeout = 10000 # ms
            print(self._query('*IDN?').strip())
        else:
            self.inst = None
        
        if reset:
            self.reset()

        
    def __del__(self):
        try:
            self.inst.clear()
            self.inst.close()
        except AttributeError:
            pass
        except Exception as e:
            print(e)

    def _query(self, cmd: str):
        """Query the PPG."""
        try:
            resp = self.inst.query(cmd)
            if resp == '\n\n':
                raise EOFError(f'Invalid command {cmd}')  # invalid command
            if resp == '\n':
                return True  # when write commands are executed
            return resp  # when query commands are executed
        except AttributeError:
            print(f'[DEBUG] {cmd}')
            return '0'
        except Exception as e:
            raise e
    
    def _check_channels(self, channels):
        """Check if channels are in the correct format and return it as array."""
        if channels is not None and not isinstance(channels, (IntegerNumber, Iterable)):
            raise ValueError('`channels` is not in the correct format')
        
        if channels is not None:
            if isinstance(channels, IntegerNumber):
                channels = np.array([channels], dtype=int)
            else:
                channels = np.array(channels, dtype=int)

            if (channels < 1).any() or (channels > self.CHANNELS).any() or channels.size > self.CHANNELS:
                channels = channels.clip(1, self.CHANNELS)[:self.CHANNELS]
                msg = f'The channels number is out of the range of the PPG3204. Setting to the limits {channels}.'
                warnings.warn(msg)
        else:
            channels = np.arange(1, self.CHANNELS+1)
        return channels


    def reset(self):
        """Reset the PPG to its default state."""
        self._query('*RST')
        return self


    def patt_len(self, length: int, CHs: int | list[int] = None):
        """Set Data Pattern Length (only relevant if type is DATA)."""
        CHs = self._check_channels(CHs)
        
        if length < self.PATT_LEN_MIN or length > self.PATT_LEN_MAX:
            warnings.warn(f"Pattern length {length} out of range. Clipping.")
            length = np.clip(length, self.PATT_LEN_MIN, self.PATT_LEN_MAX) 
        
        for ch in CHs:
            self._query(f':DIG{ch}:PATT:LENG {int(length)}')
        return self


    def get_patt_len(self, CHs: int | list[int] = None):
        """Get the current length of pattern for specified channels
        """
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':DIG{ch}:PATT:LENG?')) for ch in CHs])


    def patt_type(self, type: Literal['DATA', 'PRBS'], CHs: int | list[int] = None):
        """Set pattern type (DATA or PRBS).
        
        Parameters
        ----------
        type : str
            'DATA' or 'PRBS'.
        CHs : int or list
            Channels to configure.
        """
        CHs = self._check_channels(CHs)
        if type.upper() not in self.PATT_TYPE:
            raise ValueError(f'type must be {self.PATT_TYPE}')
        
        for ch in CHs:
            self._query(f':DIG{ch}:PATT:TYPE {type.upper()}')
        return self


    def get_patt_type(self, CHs: int | list[int] = None):
        """Get patt_type of the PPG3204 for each channels specified, can be 'DATA' or 'PRBS'

        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the patt_type.

        Returns
        -------
        patt_type : :obj:`np.ndarray`
            Every channel patt_type.
        """
        CHs = self._check_channels(CHs)
        return np.array([self._query(f':DIG{ch}:PATT:TYPE?').strip() for ch in CHs])
    

    def prbs(self, order: Literal[7, 9, 11, 15, 23, 31], CHs: int | list[int] = None):
        """Set the order of polynomial generator for PRBS patt_type.
        
        Parameters
        ----------
        order : :obj:`int` or :obj:`Array_Like(int)`
            order of the polynomial generator. Default ``7``
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to set the order.

        Raises
        ------
        ValueError
            If ``order`` is not in the correct format.

        Notes
        -----

        **PRBS pattern lengths** Independently selected for each channel.

        - :math:`2^7-1` bits. Polynomial :math:`= X^7 + X^6 + 1`
        - :math:`2^9-1` bits. Polynomial :math:`= X^9 + X^5 + 1`
        - :math:`2^{11}-1` bits. Polynomial :math:`= X^{11} + X^9 + 1`
        - :math:`2^{15}-1` bits. Polynomial :math:`= X^{15} + X^{14} + 1`
        - :math:`2^{23}-1` bits. Polynomial :math:`= X^{23} + X^{18} + 1`
        - :math:`2^{31}-1` bits. Polynomial :math:`= X^{31} + X^{28} + 1`
        """
        CHs = self._check_channels(CHs)
        if order not in self.PRBS_ORDERS:
            raise ValueError(f"Order must be one of {self.PRBS_ORDERS}")

        for ch in CHs:
            self._query(f':DIG{ch}:PATT:PLEN {order}')
        return self

    
    def get_prbs(self, CHs: int | list[int] = None):
        """Get the prbs polynomial order for each channel specified
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the order.

        Returns
        -------
        order : :obj:`np.ndarray`
            Every channel order.
        """
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':DIG{ch}:PATT:PLEN?')) for ch in CHs])
    

    def data(self, data: str | np.ndarray, start_addr: int=1, CHs: int | list[int] = None):
        """Set the data of the pattern.

        Programs the pattern data memory. Each byte of pattern data is a character (0 or 1)
        representing one bit of pattern data. The start address can be any bit location from 1 to MAX_MEMORY_LEN. MAX_MEMORY_LEN is :math:`2^{21} = 2097152` (2M) for each channel. 
        
        Parameters
        ----------
        data : :obj:`str` or :obj:`Array_Like(int)`
            Data to set to the specified channels.
        start_addr : :obj:`int`, optional
            Start address of the data to set in the pattern memory. The range is from 1 to 2^21. Default ``1``.
        CHs : :obj:`int` or :obj:`Array_Like`, optional
            Channels to set the data. If ``CHs=None`` data will be fixed in all channels.

        Raises
        ------
        ValueError
            If ``data`` is not in the correct format.

        Warns
        -----
        UserWarning
            If the length of the data is out of the range of the PPG3204.

        Examples
        --------
        In this examples we don't pass the argument ``addr_ID`` in order to print the commands output. For communication with a device this parameter is requered.

        .. code-block:: python

            >>> from opticomlib.lab import PPG3204
            >>>
            >>> ppg = PPG3204()
            >>>
            >>> ppg.set_data('000111000111', CHs=2)
            :DIG2:PATT:DATA 1,12,#212000111000111
            >>>
            >>> ppg.set_data('000111000111')
            :DIG1:PATT:DATA 1,12,#212000111000111
            :DIG2:PATT:DATA 1,12,#212000111000111
            :DIG3:PATT:DATA 1,12,#212000111000111
            :DIG4:PATT:DATA 1,12,#212000111000111
            >>>
            >>> ppg.set_data([[1,0,1,0],[0,1,0,1]], CHs=[3,4])
            :DIG3:PATT:DATA 1,4,#141010
            :DIG4:PATT:DATA 1,4,#140101

        """
        CHs = self._check_channels(CHs)

        if not isinstance(data, (str, Iterable)):
            raise ValueError('`data` is not in the correct format')
        
        if isinstance(data, str):
            data = str2array(data).astype(np.uint8)
        else:
            data = np.array(data, dtype=bool).astype(np.uint8)
        
        if any((data != 0) & (data != 1)):
            raise ValueError('`data` string must only contain 0 and 1 characters')

        if data.size > self.PATT_LEN_MAX-start_addr+1:
            msg = 'The length of the data is greater than the maximum memory length minus the start address. Setting to the nearest value.'
            warnings.warn(msg)
            data = data[:self.PATT_LEN_MAX-start_addr+1]

        # Calculate chunks (Manual page 34 says max bit count per command is 1024)
        if data.size > self.MAX_CHUNK_LEN:
            chunks = np.split(data, np.arange(self.MAX_CHUNK_LEN, data.size, self.MAX_CHUNK_LEN))
        else:
            chunks = [data]

        for ch in CHs:
            current_addr = start_addr
            
            for chunk in chunks:
                n_bits = chunk.size
                # Create IEEE-488.2 block header: #<num_digits><n_bytes><data>
                # But PED manual page 34 example uses ASCII chars 0/1: 
                # :SENS1:PATT:DATA 1,16,#2160100...
                
                # Data string "01010..."
                data_str = ''.join(chunk.astype(str))
                
                length_str = str(n_bits)
                num_digits = len(length_str)
                
                cmd = f':DIG{ch}:PATT:DATA {current_addr},{n_bits},#{num_digits}{length_str}{data_str}'
                self._query(cmd)
                current_addr += n_bits
        return self


    def get_data(self, size: int, start_addr: int=1, CHs: int | list[int] = None):
        """Get the data of the pattern for each specified channel 
        
        Parameters
        ----------
        size : :obj:`int`
            Size of the data to get from the pattern memory.
        start_addr : :obj:`int`, optional
            Start address of the data to get from the pattern memory. The range is from 1 to 2^21. Default is 1.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the data.

        Returns
        -------
        data : :obj:`np.ndarray`, shape (n_channels, n_bits)
            Data of the pattern for each channel.

        Warns
        -----
        UserWarning
            If the start address or the size is out of the range of the PPG3204.

        Raises
        ------
        ValueError
            If ``start_addr`` or ``size`` are not integers.
        """
        CHs = self._check_channels(CHs)

        if start_addr < 1 or start_addr > self.PATT_LEN_MAX:
            msg = f'`start_addr` must been between 1 and {self.PATT_LEN_MAX}. Setting to the nearest value.'
            warnings.warn(msg)
            start_addr = np.clip(start_addr, 1, self.PATT_LEN_MAX)

        if length < 1 or length > self.PATT_LEN_MAX - start_addr + 1:
            msg = f'`length` must been between 1 and (MAX_MEMORY_LEN - start_addr+1)={self.PATT_LEN_MAX - start_addr + 1}. Setting to the nearest value.'
            warnings.warn(msg)
            length = np.clip(length, 1, self.PATT_LEN_MAX - start_addr + 1)

        data_out = []
        for ch in CHs:
            current_addr = start_addr
            data_chunks = []
            remaining = length
            while remaining > 0:
                n = min(remaining, self.MAX_CHUNK_LEN)
                resp = self._query(f':DIG{ch}:PATTERN:DATA? {current_addr},{n}').strip()
                # Response format: #<k><n><data>

                if resp[0] != '#':
                    return ''
                k = int(resp[1])
                chunk = str2array(resp[k+2:-1], bool).astype(np.uint8)
                data_chunks.append(chunk)
                current_addr += n
                remaining -= n
            data_out.append(np.concatenate(data_chunks))
        return np.array(data_out)


    def bits_shift(self, bsh: int, CHs: int | list[int] = None):
        r"""Set the bits shift of the pattern
        
        Parameters
        ----------
        bsh : :obj:`int` or :obj:`Array_Like(int)`
            Bits shift to set to the specify channels.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the bits shift. If ``CHs=None`` bits shift will be fixed in all channels.
        
        Raises
        ------
        ValueError
            If ``bsh`` is not in the correct format.

        Notes
        -----
        **Pattern shift** advance or delay. This is equivalent to unlimited shifting since this range allow shifting the longest pattern to any position. 
            - **Range**: :math:`\pm(2^{30}-1)`
            - **Resolution**: 1 bit
        """
        CHs = self._check_channels(CHs)
        
        if not isinstance(bsh, (IntegerNumber, Iterable)):
            raise ValueError('`bsh` is not in the correct format')
        
        for ch in CHs:
            self._query(f':DIG{ch}:PATT:BSH {bsh}')
        return self


    def get_bits_shift(self, CHs: int | list[int] = None):
        """Get the bits shift of the pattern for each specified channel
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the bits shift.

        Returns
        -------
        bsh : :obj:`np.ndarray`
            Every channel bits shift.
        """
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':DIG{ch}:PATT:BSH?')) for ch in CHs])


    def output(self, state: Literal[0 , 1, 'ON', 'OFF'], CHs: int | list[int] = None):
        """Enable or disable the output of the channels
        
        Parameters
        ----------
        state : :obj:`int` {0, 1} or :obj:`str` {'ON', 'OFF'}
            State to set for the channels ('ON' to enable, 'OFF' to disable).
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the output state. If ``CHs=None`` all channels will be affected.
        """
        CHs = self._check_channels(CHs)
        
        if isinstance(state, IntegerNumber):
            state = 'ON' if state == 1 else 'OFF'
        elif isinstance(state, str):
            state = state.upper()
        
        for ch in CHs:
            self._query(f':OUTP{ch} {state}')

    def get_output(self, CHs: int | list[int] = None):
        """Get the output state of the channels
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to get the output state. If ``CHs=None`` all channels will be queried.
        
        Returns
        -------
        state : :obj:`np.ndarray`
            Every channel output state ('ON' or 'OFF').
        """
        CHs = self._check_channels(CHs)
        return np.array([self._query(f':OUTP{ch}?').strip() for ch in CHs])


    def data_rate(self, value: float):
        r"""Set the bit rate of the pattern

        - *Range*: 1.5 GHz to 32 GHz
        - *Resolution*: 10 kb/s
        - *Accuracy*: :math:`\pm 5` ppm
        
        Parameters
        ----------
        value : :obj:`float`
            Bit Rate of the pattern in bits/s.
        """
        if value < self.FREQ_MIN or value > self.FREQ_MAX:
            value = np.clip(value, self.FREQ_MIN, self.FREQ_MAX)
            msg = f'The frequency is out of the range of the PPG3204. Setting to the limits {value:.2e} Hz.'
            warnings.warn(msg)

        self._query(f':FREQ {value:.5e}')
        return self
    

    def get_data_rate(self):
        """Get the frequency of the pattern.
        
        Returns
        -------
        freq: :obj:`float`
            Bit Rate of the pattern in bits/s.
        """
        return float(self._query(':FREQ?'))
    

    def skew(self, skew: float, CHs: int | list[int] = None):
        """Set the skew of the channels
        
        The channel skew is the timing of the data output. 

        - *Range*: -25 to 25 ps
        - *Resolution*: 0.1 ps
        
        Parameters
        ----------
        skew : :obj:`float` or :obj:`Array_Like(float)`
            Skew to set to the specify channels
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the skew. If ``CHs=None`` skew will be fixed in all channels.

        Raises
        ------
        ValueError
            If ``skew`` is not in the correct format.
        """
        CHs = self._check_channels(CHs)

        if not isinstance(skew, (RealNumber, Iterable)):
            raise ValueError('`skew` is not in the correct format')
        

        if skew < self.MIN_SKEW or skew > self.MAX_SKEW:
            skew = skew.clip(self.MIN_SKEW, self.MAX_SKEW)
            msg=f'The skew is out of the range of the PPG3204. Setting to the limits {skew}.'
            warnings.warn(msg)
        
        for ch in CHs:
            self._query(f':SKEW{ch} {skew}')
        return self


    def get_skew(self, CHs: int | list[int] = None):
        """Get the skew of the channels
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the skew.
        
        Returns
        -------
        skew : :obj:`np.ndarray`
            Every channel skew.
        """
        CHs = self._check_channels(CHs)
        return np.array([float(self._query(f':SKEW{ch}?')) for ch in CHs])


    def amplitude(self, value: float | list[float], CHs: int | list[int] = None):
        """Set the peak-to-peak output voltage (in mV).
        
        Parameters
        ----------
        value : :obj:`float` or :obj:`Array_Like`
            Amplitude to set to the specify channels
        CHs : :obj:`int` or :obj:`Array_Like`, optional
            Channels to set the amplitude. If ``CHs=None`` amplitude will be fixed in all channels.
        """
        CHs = self._check_channels(CHs)
        
        if not isinstance(value, (RealNumber, Iterable)):
            raise ValueError('`value` is not in the correct format')
        
        value = float(value)*1e-3

        if value < self.AMPLITUDE_MIN or value > self.AMPLITUDE_MAX:
            value = np.clip(value, self.AMPLITUDE_MIN, self.AMPLITUDE_MAX) 
            msg = f'The amplitude is out of the range of the PPG3204. Setting to the limits {value:.2f}.'
            warnings.warn(msg)
        
        for ch in CHs:
            self._query(f':VOLT{ch}:POS {value:.1f}v')
        return self

    def get_amplitude(self, CHs: int | list[int] = None):
        """Get the peak-to-peak output voltage (in mV).
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the amplitude.

        Returns
        -------
        Vout : :obj:`np.ndarray`
            Every channel output voltage.
        """
        CHs = self._check_channels(CHs)
        return np.array([float(self._query(f':VOLT{ch}:POS?'))*1e3 for ch in CHs])


    def offset(self, value: float, CHs: int | list[int] = None):
        """Set offset voltage (in mV)
        
        Parameters
        ----------
        value : :obj:`float` or :obj:`Array_Like(float)`
            Offset to set to the specify channels
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the offset. If ``CHs=None`` offset will be fixed in all channels.

        Raises
        ------
        ValueError
            If ``value`` is not in the correct format.

        Warns
        -----
        UserWarning
            If the offset is out of the range of the PPG3204.

        Notes
        -----
        
        **Offset adjust** relative to nominal position. 
            - **Range**: -2000 to 3000 mV
        """
        CHs = self._check_channels(CHs)

        if not isinstance(value, (RealNumber, Iterable)):
            raise ValueError('`value` is not in the correct format')
        
        value = float(value)*1e-3
        
        if value < self.OFFSET_MIN or value > self.OFFSET_MAX:
            value = value.clip(self.OFFSET_MIN, self.OFFSET_MAX)
            msg = f'The offset is out of the range of the PPG3204. Setting to the limits {value:.2f}.'
            warnings.warn(msg)

        for ch in CHs:
            if value < 0:
                self._query(f':VOLT{ch}:NEG:OFFS {value:.1f}v')
            else:
                self._query(f':VOLT{ch}:POS:OFFS {value:.1f}v')
        return self


    def get_offset(self, CHs: int | list[int] = None):
        """Get the offset voltage (in mV) 
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the offset.
        
        Returns
        -------
        offset : :obj:`np.ndarray`
            Every channel offset.
        """
        CHs = self._check_channels(CHs)
        return np.array([float(self._query(f':VOLT{ch}:OFFS?'))*1e3 for ch in CHs])


    def __call__(self, 
               data_rate: float = None, 
               patt_len: int | list[int] = None, 
               amplitude: float | list[float] = None,
               offset: float | list[float] = None,
               bsh: int | list[int] = None, 
               skew: float | list[float] = None,
               patt_type: Literal['DATA', 'PRBS'] = None, 
               prbs: int | list[int] = None, 
               data: np.ndarray | list[np.ndarray] = None,
               output: Literal[0, 1, 'ON', 'OFF'] = None,
               CHs: int | list[int] = None):
        """ Configure the PPG3204 with the specified parameters for specified channels.

        Parameters
        ----------
        data_rate : :obj:`float`, optional
            Frequency of the pattern in Hz. The range is from 1.5 GHz to 32 GHz.
        patt_len : :obj:`int` or :obj:`Array_Like(int)`, optional
            Pattern length for every channel specified in ``CHs``.
        amplitude : :obj:`float` or :obj:`Array_Like(float)`, optional
            Amplitude to set to the specify channels
        offset : :obj:`float` or :obj:`Array_Like(float)`, optional
            Offset to set to the specify channels
        bsh : :obj:`int` or :obj:`Array_Like(int)`, optional
            Bits shift to set to the specify channels
        skew : :obj:`float` or :obj:`Array_Like(float)`, optional
            Skew to set to the specify channels
        patt_type : :obj:`str`, optional
            Work patt_type of the PPG, ``"DATA"`` or ```"PRBS"``. Default ``"PRBS"``
        prbs : :obj:`int` or :obj:`Array_Like(int)`, optional
            order of the polynomial generator. If ``patt_type='PRBS'``.
        data : :obj:`np.ndarray` or :obj:`Array_Like(np.ndarray)`, optional
            Data to set to the specify channels. If ``patt_type='DATA'``.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the configuration.

        Examples
        --------
        In this examples we don't pass the argument ``addr_ID`` in order to print the commands output. For communication with a device this parameter is required.

        .. code-block:: python

            >>> from opticomlib.lab import PPG3204
            >>> 
            >>> ppg = PPG3204()
            >>> ppg(data_rate=10e9, patt_len=1000, amplitude=1.5, offset=0.5, bsh=10, skew=0.5e-12, patt_type='PRBS', prbs=7, CHs=2)
            :FREQ 1.0e+10
            :DIG2:PATT:LENG 1000
            :VOLT2:POS 1.5v
            :VOLT2:POS:OFFS 0.5v
            :DIG2:PATT:BSH 10
            :SKEW2 5e-13
            :DIG2:PATT:TYPE PRBS
            :DIG2:PATT:PLEN 7
        """
        if data_rate is not None:
            self.data_rate(data_rate)

        if patt_len is not None:
            self.patt_len(patt_len, CHs)
        
        if amplitude is not None:
            self.amplitude(amplitude, CHs)

        if offset is not None:
            self.offset(offset, CHs)

        if bsh is not None:
            self.bits_shift(bsh, CHs)

        if skew is not None:
            self.skew(skew, CHs)

        if patt_type is not None:
            self.patt_type(patt_type, CHs)

        if prbs is not None and patt_type == 'PRBS':
            self.prbs(prbs, CHs)
        
        if data is not None and patt_type == 'DATA':
            self.data(data, start_addr=1, CHs=CHs)

        if output is not None:
            self.output(output, CHs)

        print('Done')
        return self
    
    def get_metadata(self, ch : int=1):
        """Retrieve a summary of the current PPG configuration for the specified channel as a dictionary."""
        metadata = {
            'PATT_TYPE': self.get_patt_type(ch)[0],
            'PATT_LEN': self.get_patt_len(ch)[0],
            'PRBS': self.get_prbs(ch)[0],
            'AMPLITUDE': self.get_amplitude(ch)[0],
            'OFFSET': self.get_offset(ch)[0],
            'OUTPUT': self.get_output(ch)[0],
            'DATA_RATE': self.get_data_rate(),
            'SKEW': self.get_skew(ch)[0],
            'BITS_SHIFT': self.get_bits_shift(ch)[0],
        }
        return metadata
    
    def print_setup(self, ch: int = None):
        """ Print the current configuration of the PPG3204 for a specified channel."""
        metadata = self.get_metadata(ch)
        print("=== PPG3204 SETUP ===")
        for key, value in metadata.items():
            print(f"{1*' ' + key + (11 - len(key))*' '}: {value}")
        print("======================")

    
    def setup(self, 
            data_rate: float = None, 
            patt_type: Literal['DATA', 'PRBS'] = None, 
            patt_len: int | list[int] = None, 
            amplitude: float | list[float] = None,
            offset: float | list[float] = None,
            bsh: int | list[int] = None, 
            skew: float | list[float] = None,
            prbs: int | list[int] = None, 
            data: np.ndarray | list[np.ndarray] = None,
            output: Literal[0, 1, 'ON', 'OFF'] = None,
            CHs: int | list[int] = None):
        """ Configure the PPG3204 with the specified parameters for specified channels.
        
        Parameters
        ----------
        data_rate : :obj:`float`, optional
            Frequency of the pattern in Hz. The range is from 1.5 GHz to 32 GHz.
        patt_len : :obj:`int` or :obj:`Array_Like(int)`, optional
            Pattern length for every channel specified in ``CHs``.
        amplitude : :obj:`float` or :obj:`Array_Like(float)`, optional
            Amplitude to set to the specify channels
        offset : :obj:`float` or :obj:`Array_Like(float)`, optional
            Offset to set to the specify channels
        bsh : :obj:`int` or :obj:`Array_Like(int)`, optional
            Bits shift to set to the specify channels
        skew : :obj:`float` or :obj:`Array_Like(float)`, optional
            Skew to set to the specify channels
        patt_type : :obj:`str`, optional
            Work patt_type of the PPG.
        prbs : :obj:`int` or :obj:`Array_Like(int)`, optional
            order of the polynomial generator. If ``patt_type='PRBS'``.
        data : :obj:`np.ndarray` or :obj:`Array_Like(np.ndarray)`, optional
            Data to set to the specify channels. If ``patt_type='DATA'``.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the configuration.

        Examples
        --------
        In this examples we don't pass the argument ``addr_ID`` in order to print the commands output. For communication with a device this parameter is required.

        .. code-block:: python

            >>> from opticomlib.lab import PPG3204
            >>> 
            >>> ppg = PPG3204()
            >>> ppg(data_rate=10e9, patt_len=1000, amplitude=1.5, offset=0.5, bsh=10, skew=0.5e-12, patt_type='PRBS', prbs=7, CHs=2)
            :FREQ 1.0e+10
            :DIG2:PATT:LENG 1000
            :VOLT2:POS 1.5v
            :VOLT2:POS:OFFS 0.5v
            :DIG2:PATT:BSH 10
            :SKEW2 5e-13
            :DIG2:PATT:TYPE PRBS
            :DIG2:PATT:PLEN 7
        """
        self.__call__(data_rate, patt_len, amplitude, offset, bsh, skew, patt_type, prbs, data, output, CHs)







class PED4002():
    """
    **Tektronix PED3200 / PED4000 Series Programmable Error Detector**

    High-performance programmable error detector (up to 32 Gb/s for PED3200, 40 Gb/s for PED4000).
    This class mirrors the remote programming command set described in the user manual
    `PED3200-PED4000-Programmable-Error-Detector-User-Manual-077109501.pdf`.

    .. rubric:: Attributes
    .. autosummary::

        ~PED4002.inst
        CHANNELS
        PATT_LEN_MIN
        PATT_LEN_MAX_1CH
        PATT_LEN_MAX_2CH
        CLK_DELAY_MIN
        CLK_DELAY_MAX
        EYE_THRESH_MIN
        EYE_THRESH_MAX
        SYNC_THRESH_MIN
        SYNC_THRESH_MAX
        PATT_TYPE
        PRBS_ORDERS
        MAX_CHUNK_LEN

    .. rubric:: Main methods
    .. autosummary::

        __init__
        __call__
        reset
        patt_len
        get_patt_len
        patt_type
        get_patt_type
        prbs
        get_prbs
        data
        get_data
        sync
        is_sync
        sync_threshold
        get_sync_threshold
        center_offset
        offset
        get_offset
        get_voltage_edges
        center_delay
        delay
        get_delay
        get_time_edges
        eye_threshold
        get_eye_threshold
        is_running
        run
        stop
        get_ber
        get_error_count
        get_bit_count
        get_frequency
        setup
        get_metadata
        print_setup
    """
    CHANNELS = 2 
    """Maximum number of channels (Model dependent)."""
    PATT_LEN_MIN = 2
    """Pattern length minimum value."""
    PATT_LEN_MAX_1CH = 4_194_304
    """Pattern length maximum value for single channel config (4 Mbit)."""
    PATT_LEN_MAX_2CH = 2_097_152
    """Pattern length maximum value per channel for 2-ch config (2 Mbit)."""
    CLK_DELAY_MIN = -50
    """Minimum clock delay (-50 ps)."""
    CLK_DELAY_MAX = 50
    """Maximum clock delay (+50 ps)."""
    EYE_THRESH_MIN = 1e-11
    """Minimum Eye Edge BER Threshold."""
    EYE_THRESH_MAX = 1e-1
    """Maximum Eye Edge BER Threshold."""
    SYNC_THRESH_MIN = 1e-8
    """Minimum Synchronization BER Threshold."""
    SYNC_THRESH_MAX = 1e-1
    """Maximum Synchronization BER Threshold."""
    PATT_TYPE = ['DATA', 'PRBS']
    """Mode of the error detector."""
    PRBS_ORDERS = [7, 9, 11, 15, 23, 31]
    """Supported PRBS polynomial orders."""
    MAX_CHUNK_LEN = 1024
    """Maximum length of the data block to write in a single command."""

    def __init__(self, addr_ID: str = None, reset: bool = True):
        """ Initialize the PED.
        
        Parameters
        ----------
        addr_ID : :obj:`str`, optional
            VISA resource of the PED. Default is None (Debug mode).
        reset : :obj:`bool`, optional
            If True, reset the instrument to factory defaults upon connection. Default is True.
        """
        if addr_ID:
            self.inst = visa.ResourceManager().open_resource(addr_ID)
            self.inst.timeout = 10000  # ms
            print(self._query('*IDN?'))
        else:
            self.inst = None
            self._query('*IDN?')

        if reset:
            self.reset()

    def __del__(self):
        try:
            if self.inst:
                self.inst.close()
        except Exception as e:
            print(e)

    def _query(self, cmd: str):
        """Internal query/write helper (prints command if no connection)."""
        try:
            if self.inst:
                resp = self.inst.query(cmd)
                
                if resp == '\n\n':
                    raise EOFError(f'Invalid command {cmd}')  # invalid command
                if resp == '\n':
                    return True  # when write commands are executed
                return resp  # when query commands are executed
            else:
                print(f'[DEBUG] {cmd}')
                return '0'
        except Exception as e:
            print(f"Error sending command '{cmd}': {e}")
            raise e

    def _check_channels(self, channels):
        """Check if channels are correct and return as array."""
        if channels is not None and not isinstance(channels, (IntegerNumber, Iterable)):
            raise ValueError('`channels` is not in the correct format')
        
        if channels is not None:
            if isinstance(channels, IntegerNumber):
                channels = np.array([channels], dtype=int)
            else:
                channels = np.array(channels, dtype=int)

            if (channels < 1).any() or (channels > self.CHANNELS).any():
                channels = channels.clip(1, self.CHANNELS)
                warnings.warn(f'Channels clipped to limits {channels}.')
        else:
            channels = np.arange(1, self.CHANNELS+1)
        return channels
    
    def _get_nodes(self, channel: int):
        """
        Returns the SCPI nodes for Data and Clock based on channel.
        According to Manual Page 18/34:
        Ch1 Data -> SENSe1, Ch1 Clock -> SENSe2/INPut2
        Ch2 Data -> SENSe3, Ch2 Clock -> SENSe4/INPut4
        """
        data_node = 1 + 2 * (channel - 1)
        clock_node = 2 + 2 * (channel - 1)
        return data_node, clock_node

    # ==============================================================
    # Basic instrument control
    # ==============================================================

    def reset(self):
        """Reset the PED to default settings (``*RST``)."""
        self._query('*RST')
        # Wait for operation complete
        self._query('*OPC?')
        return self

    # ==============================================================
    # Pattern configuration (same for both data channels)
    # ==============================================================
    
    def patt_len(self, length: int, CHs: int | list[int] = None):
        """Set Data Pattern Length (only relevant if type is DATA)."""
        CHs = self._check_channels(CHs)
        
        # Simple limit check (assuming 2CH mode for safety if generic)
        limit = self.PATT_LEN_MAX_2CH if self.CHANNELS > 1 else self.PATT_LEN_MAX_1CH

        if length < self.PATT_LEN_MIN or length > limit:
            warnings.warn(f"Pattern length {length} out of range. Clipping.")
            length = np.clip(length, self.PATT_LEN_MIN, limit)

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:PATT:LENG {int(length)}')
        return self

    
    def get_patt_len(self, CHs: int | list[int] = None):
        """Get the current length of pattern for specified channels
        """
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':SENS{self._get_nodes(ch)[0]}:PATT:LENG?')) for ch in CHs])
    

    def patt_type(self, type: Literal['DATA', 'PRBS'], CHs: int | list[int] = None):
        """Set pattern type (DATA or PRBS).
        
        Parameters
        ----------
        type : str
            'DATA' or 'PRBS'.
        CHs : int or list
            Channels to configure.
        """
        CHs = self._check_channels(CHs)
        if type.upper() not in self.PATT_TYPE:
            raise ValueError(f'type must be {self.PATT_TYPE}')
        
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:PATT:TYPE {type.upper()}')
        return self
    
    def get_patt_type(self, CHs: int | list[int]=None):
        """Get current pattern type."""
        CHs = self._check_channels(CHs)
        results = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            results.append(self._query(f':SENS{d_node}:PATT:TYPE?').strip())
        return np.array(results)

    def prbs(self, order: Literal[7, 9, 11, 15, 23, 31], CHs: int | list[int] = None):
        """Set PRBS Polynomial order (2^N - 1).
        
        Parameters
        ----------
        order : int
            One of [7, 9, 11, 15, 23, 31].
        """
        CHs = self._check_channels(CHs)
        if order not in self.PRBS_ORDERS:
            raise ValueError(f"Order must be one of {self.PRBS_ORDERS}")

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:PATT:PLEN {order}')
        return self

    def get_prbs(self, CHs: int | list[int]=None):
        """Get current PRBS order."""
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':SENS{self._get_nodes(ch)[0]}:PATT:PLEN?')) for ch in CHs])
    

    def data(self, data: str | np.ndarray, start_addr: int=1, CHs: int | list[int] = None):
        """Program the user pattern data.
        
        Parameters
        ----------
        data : str or array
            The binary data (e.g. "010110").
        start_addr : int
            Memory start address (1-based).
        CHs : int or list
            Channels to program.
        """
        CHs = self._check_channels(CHs)

        if not isinstance(data, (str, Iterable)):
            raise ValueError('`data` is not in the correct format')
        
        if isinstance(data, str):
            data = str2array(data).astype(np.uint8)
        else:
            data = np.array(data, dtype=bool).astype(np.uint8)
        
        if any((data != 0) & (data != 1)):
            raise ValueError('`data` string must only contain 0 and 1 characters')

        limit = self.PATT_LEN_MAX_2CH if self.CHANNELS > 1 else self.PATT_LEN_MAX_1CH

        if data.size > limit-start_addr+1:
            msg = 'The length of the data is greater than the maximum memory length minus the start address. Setting to the nearest value.'
            warnings.warn(msg)
            data = data[:limit-start_addr+1]

        # Calculate chunks (Manual page 34 says max bit count per command is 1024)
        if data.size > self.MAX_CHUNK_LEN:
            chunks = np.split(data, np.arange(self.MAX_CHUNK_LEN, data.size, self.MAX_CHUNK_LEN))
        else:
            chunks = [data]

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            current_addr = start_addr
            
            for chunk in chunks:
                n_bits = chunk.size
                # Create IEEE-488.2 block header: #<num_digits><n_bytes><data>
                # But PED manual page 34 example uses ASCII chars 0/1: 
                # :SENS1:PATT:DATA 1,16,#2160100...
                
                # Data string "01010..."
                data_str = ''.join(chunk.astype(str))
                
                length_str = str(n_bits)
                num_digits = len(length_str)
                
                cmd = f':SENS{d_node}:PATT:DATA {current_addr},{n_bits},#{num_digits}{length_str}{data_str}'
                self._query(cmd)
                current_addr += n_bits
        return self

    def get_data(self, length: int, start_addr: int=1, CHs: int | list[int] = None):
        """
        Retrieve binary pattern data as numpy bool array.
        """
        CHs = self._check_channels(CHs)

        limit = self.PATT_LEN_MAX_2CH if self.CHANNELS > 1 else self.PATT_LEN_MAX_1CH

        if start_addr < 1 or start_addr > limit:
            msg = f'`start_addr` must been between 1 and {limit}. Setting to the nearest value.'
            warnings.warn(msg)
            start_addr = np.clip(start_addr, 1, limit)

        if length < 1 or length > limit - start_addr + 1:
            msg = f'`length` must been between 1 and (MAX_MEMORY_LEN - start_addr+1)={limit - start_addr + 1}. Setting to the nearest value.'
            warnings.warn(msg)
            length = np.clip(length, 1, limit - start_addr + 1)

        data_out = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            current_addr = start_addr
            data_chunks = []
            remaining = length
            while remaining > 0:
                n = min(remaining, self.MAX_CHUNK_LEN)
                resp = self._query(f':SENSE{d_node}:PATTERN:DATA? {current_addr},{n}').strip()
                # Response format: #<k><n><data>

                if resp[0] != '#':
                    return ''
                k = int(resp[1])
                chunk = str2array(resp[k+2:-1], bool).astype(np.uint8)
                data_chunks.append(chunk)
                current_addr += n
                remaining -= n
            data_out.append(np.concatenate(data_chunks))
        return np.array(data_out)
    
    # =================================================================
    # CLOCK & ALIGNMENT SETTINGS
    # =================================================================

    def sync(self, CHs: int | list[int] = None, wait=True):
        """Initiate pattern synchronization."""
        CHs = self._check_channels(CHs)

        resp = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:SYNC:EXEC ONCE')
            if wait:
                while True:
                    status = self._query(f':SENS{d_node}:SYNC:EXEC?').strip()
                    if status == 'BUSY':
                        time.sleep(0.2)
                    else:
                        if status != 'OK':
                            warnings.warn(f"Sync failed on CH{ch}: {status}")
                            resp.append(False)
                        else:
                            resp.append(True)
                        break
        return np.array(resp) if wait else self
                        
    
    def is_sync(self, CHs: int | list[int] = None):
        """Check if the specified channels are synchronized."""
        CHs = self._check_channels(CHs)
        sync_status = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            status = self._query(f':SENS{d_node}:SYNC:STAT?').strip()
            sync_status.append(status == 'SYNC')
        return np.array(sync_status)
    
    def sync_threshold(self, ber: float, CHs: int | list[int] = None):
        """Programs the synchronization BER threshold. This is the maximum BER value for which a synchronization is considered successful. Also, in auto sync mode, the current BER is monitored and compared to this threshold. The threshold may range from 10-1 to 10-8 in decade steps.
        Synchronization will succeed only if the BER of the system is less than the sync BER threshold.
        
        Parameters
        ----------
        ber : float
            BER threshold between 1e-8 and 1e-1.
        CHs : int or list
            Channels to configure.
        """
        CHs = self._check_channels(CHs)
        
        if ber < self.SYNC_THRESH_MIN or ber > self.SYNC_THRESH_MAX:
            warnings.warn(f"BER threshold {ber} out of range.")
            ber = np.clip(ber, self.SYNC_THRESH_MIN, self.SYNC_THRESH_MAX)

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:SYNC:THR {ber:.1e}')
        return self
    
    def get_sync_threshold(self, CHs: int | list[int] = None):
        """Get current synchronization BER threshold."""
        CHs = self._check_channels(CHs)
        bers = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            resp = self._query(f':SENS{d_node}:SYNC:THR?').strip()
            bers.append(float(resp))
        return np.array(bers)
    
    # ==============================================================
    # Eye & decision threshold
    # ==============================================================

    def center_offset(self, CHs: int | list[int] = None, wait=True):
        """Initiates the center offset process.
        
        The center offset process can take a significant amount of time. It uses the EYE EDGE BER THRESHOLD to determine the eye edges during the process. Lower EYE EDGE BER THRESHOLD values take more time, as do data patterns (vs PRBS) and longer data pattern lengths.
        """
        CHs = self._check_channels(CHs)
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:EYE:OCENter ONCE') # begins the center offset process
            if wait:
                while True:
                    status = self._query(f':SENS{d_node}:EYE:OCENter?').strip()
                    if status == 'BUSY':
                        time.sleep(0.2)
                    else:
                        if status != 'OK':
                            warnings.warn(f"Sync failed on CH{ch}: {status}")
                        break
        return self

    def offset(self, offset: float, CHs: int | list[int] = None):
        """Set decision threshold offset (in mV). Range -300 mV to +300 mV.
        
        Programs the data offset voltage. The default value of 0 is normally good for 50% duty input signals.
        """
        CHs = self._check_channels(CHs)

        if offset < -300 or offset > 300:
            warnings.warn(f"Offset {offset} mV out of range.")
            offset = np.clip(offset, -300, 300)

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:EYE:OFFS {offset*1e-3}')
        return self
    
    def get_offset(self, CHs: int | list[int] = None):
        """Get current decision threshold offset (in mV)."""
        CHs = self._check_channels(CHs)
        offsets = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            resp = self._query(f':SENS{d_node}:EYE:OFFS?').strip()
            offsets.append(float(resp)*1e3)
        return np.array(offsets)
    
    def center_delay(self, CHs: int | list[int] = None, wait=True):
        """Initiates the center clock delay process.
        
        The center clock delay process can take a significant amount of time. It uses the EYE EDGE BER THRESHOLD to determine the eye edges during the process. Lower EYE EDGE BER THRESHOLD values take more time, as do data patterns (vs PRBS) and longer data pattern lengths.
        """
        CHs = self._check_channels(CHs)
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:EYE:TCENter ONCE') # begins the center offset process
            if wait:
                while True:
                    status = self._query(f':SENS{d_node}:EYE:TCENter?').strip()
                    if status == 'BUSY':
                        time.sleep(0.2)
                    else:
                        if status != 'OK':
                            warnings.warn(f"Sync failed on CH{ch}: {status}")
                        break
        return self
    
    def delay(self, delay: float, CHs: int | list[int] = None):
        """Set Clock to Data delay (in ps).
        
        Range: +/- 50 ps.
        """
        CHs = self._check_channels(CHs)
        
        if delay < self.CLK_DELAY_MIN or delay > self.CLK_DELAY_MAX:
            warnings.warn(f"Delay {delay} out of range.")
            delay = np.clip(delay, self.CLK_DELAY_MIN, self.CLK_DELAY_MAX)

        for ch in CHs:
            _, c_node = self._get_nodes(ch) # Uses Input Node (2 or 4)
            # Manual page 23: :INPut[2|4]:DELay
            self._query(f':INP{c_node}:DEL {delay}ps')
        return self
    
    def get_delay(self, CHs: int | list[int] = None):
        """Get current Clock to Data delay (in picoseconds)."""
        CHs = self._check_channels(CHs)
        delays = []
        for ch in CHs:
            _, c_node = self._get_nodes(ch) # Uses Input Node (2 or 4)
            resp = self._query(f':INP{c_node}:DEL?').strip()
            delays.append(float(resp)*1e12)
        return np.array(delays)

    def get_time_edges(self, CHs: int | list[int] = None):
        """Get Eye Time Edges (in secons).
        
        Queries the left or right (time axis) eye edges as determined during the most recent automatic process that sets the horizontal sampling point. This includes the Center Clock Delay and Auto Align processes. If those processes have not been run or failed on the most recent attempt, the return value will be 9.91e37. (NaN)
        """
        CHs = self._check_channels(CHs)
        edges = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            t_left = self._query(f':SENS{d_node}:EYE:TEDGE? 1').strip()
            t_right = self._query(f':SENS{d_node}:EYE:TEDGE? 2').strip()
        
            t_left = float(t_left)
            t_right = float(t_right)
            if t_left > 9e37:
                t_left = np.nan
            if t_right > 9e37:
                t_right = np.nan   

            t_edges = (t_left, t_right)
            edges.append(t_edges)
        return np.array(edges)
    
    def eye_threshold(self, ber: float, CHs: int | list[int] = None):
        """Set Eye Edge BER Threshold.
        
        Programs the eye edge BER threshold. This is the maximum BER value for which an eye edge is considered valid during automatic processes that determine eye edges. The threshold may range from 10-1 to 10-11 in decade steps.
        """
        CHs = self._check_channels(CHs)
        
        if ber < self.EYE_THRESH_MIN or ber > self.EYE_THRESH_MAX:
            warnings.warn(f"BER threshold {ber} out of range.")
            ber = np.clip(ber, self.EYE_THRESH_MIN, self.EYE_THRESH_MAX)

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:EYE:THR {ber}')
        return self
    
    def get_eye_threshold(self, CHs: int | list[int] = None):
        """Get current Eye Edge BER Threshold."""
        CHs = self._check_channels(CHs)
        bers = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            resp = self._query(f':SENS{d_node}:EYE:THR?').strip()
            bers.append(float(resp))
        return np.array(bers)
    
    def get_voltage_edges(self, CHs: int | list[int] = None):
        """Get Eye Voltage Edges (in Volts).
        
        Queries the upper and lower (voltage axis) eye edges as determined during the most recent automatic process that sets the vertical sampling point. This includes the Center Offset and Auto Align processes. If those processes have not been run or failed on the most recent attempt, the return value will be 9.91e37. (NaN)
        """
        CHs = self._check_channels(CHs)
        edges = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            v_up = self._query(f':SENS{d_node}:EYE:VEDG? 2').strip()
            v_down = self._query(f':SENS{d_node}:EYE:VEDG? 1').strip()
            
            v_up = float(v_up)
            v_down = float(v_down)
            if v_up > 9e37:
                v_up = np.nan
            if v_down > 9e37:
                v_down = np.nan

            v_edges = (v_down, v_up)
            edges.append(v_edges)
        return np.array(edges)*1e3  # in mV

    # ==============================================================
    # Gating
    # ==============================================================
    
    def is_running(self, CHs: int | list[int] = None):
        """Check if error detection is running (gating enabled)."""
        CHs = self._check_channels(CHs)
        states = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            resp = self._query(f':SENS{d_node}:GATE:STATE?').strip()
            states.append(resp == 'ON')
        return np.array(states)

    def run(self, CHs: int | list[int] = None):
        """Start error detection (enable gating)."""
        CHs = self._check_channels(CHs)

        if not self.is_sync(CHs).all():
            if self.sync(CHs, wait=True).all() is False:
                warnings.warn("Cannot start error detection: synchronization failed on one or more channels.")
                return self
            
        if self.is_running(CHs).all():
            warnings.warn("Error detection already running.")
            return self  # already running

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:GATE:STATE ON')
        return self
    
    def stop(self, CHs: int | list[int] = None):
        """Stop error detection (disable gating)."""
        CHs = self._check_channels(CHs)

        if not self.is_running(CHs).any():
            warnings.warn("Error detection already stopped.")
            return self  # already stopped

        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            self._query(f':SENS{d_node}:GATE:STATE OFF')
        return self

    # ==============================================================
    # Measurement results
    # ==============================================================

    def get_ber(self, CHs: int | list[int] = None):
        """Get current Bit Error Ratio.
        
        Returns NaN if not synced or valid.
        """
        CHs = self._check_channels(CHs)
        res = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            # Page 22: :FETC:ERAT?
            val_str = self._query(f':FETC:SENS{d_node}:ERAT?').strip()
            val = float(val_str)
            if val > 9e36: val = np.nan
            res.append(val)
        return np.array(res)
        

    def get_error_count(self, CHs: int | list[int] = None):
        """Get total error count."""
        CHs = self._check_channels(CHs)
        res = []
        for ch in CHs:
            d_node, _ = self._get_nodes(ch)
            val_str = self._query(f':FETC:SENS{d_node}:ECO?').strip()
            val = float(val_str)
            if val > 9e36: val = np.nan
            res.append(val)
        return np.array(res)
    

    def get_bit_count(self, CHs: int | list[int] = None):
        """Get total bit count."""
        CHs = self._check_channels(CHs)
        res = []
        for ch in CHs:
            _, c_node = self._get_nodes(ch)
            # Page 22: :FETCh:SENSe[2|4]:BCOunt? (Associated with clock node)
            val_str = self._query(f':FETC:SENS{c_node}:BCO?').strip()
            val = float(val_str)
            if val > 9e36: val = np.nan
            res.append(val)
        return np.array(res)
        

    def get_frequency(self, CHs: int | list[int] = None):
        """Queries the frequency measured at the clock input. 
        
        This value is valid only if the error detector is synchronized. The return value is in Hz.
        """
        CHs = self._check_channels(CHs)
        res = []
        for ch in CHs:
            _, c_node = self._get_nodes(ch)
            # Page 37: :SENSe[2|4]:FREQuency?
            val_str = self._query(f':SENS{c_node}:FREQ?').strip()
            val = float(val_str)
            if val > 9e36: val = np.nan
            res.append(val)
        return np.array(res)
        
       

    # ==============================================================
    # Convenience configuration callable
    # ==============================================================
    
    def setup(self,
                 patt_type: Literal['DATA', 'PRBS']=None,
                 patt_len: int=None,
                 prbs: Literal[7, 9, 11, 15, 23, 31]=None,
                 data: str | np.ndarray=None,
                 eye_threshold: float=None,
                 center_delay: bool=False,
                 center_offset: bool=False,
                 offset_mV: float=None,
                 delay_ps: float=None,
                 sync_threshold: float=None,
                 sync: bool=None,
                 run: bool=None,
                 stop: bool=None,
                 CHs: int | list[int] = None,
                 ):
        """
        Configure the PED with the specified parameters in a sequential and logical order.

        This method allows setting up the physical layer (alignment), the logical layer (pattern/sync), 
        and the measurement state (run/stop) in a single function call.

        Parameters
        ----------
        patt_type : :obj:`str` {'DATA', 'PRBS'}, optional
            Pattern type expected by the error detector.
        patt_len : :obj:`int`, optional
            Length of the pattern (only relevant if ``patt_type='DATA'``).
        prbs : :obj:`int`, optional
            Polynomial order for PRBS patterns (e.g., 7, 15, 31). Only used if ``patt_type='PRBS'``.
        data : :obj:`str` or :obj:`np.ndarray`, optional
            Binary data sequence to be programmed into memory. Only used if ``patt_type='DATA'``.
        eye_threshold : :obj:`float`, optional
            BER threshold used during the auto-alignment process (``center_delay`` / ``center_offset``) 
            to detect the edges of the eye. Default is usually 1e-3.
        center_delay : :obj:`bool`, optional
            If True, initiates the **Auto-Align Clock Delay** process. The instrument scans the horizontal 
            axis to find the center of the data eye (optimal sampling time).
        center_offset : :obj:`bool`, optional
            If True, initiates the **Auto-Align Voltage Offset** process. The instrument scans the vertical 
            axis to find the center of the data eye (optimal sampling voltage).
        offset_mV : :obj:`float`, optional
            Manually sets the decision threshold voltage offset in millivolts (-300 to +300 mV).
        delay_ps : :obj:`float`, optional
            Manually sets the clock-to-data delay in picoseconds (-50 to +50 ps).
        sync_threshold : :obj:`float`, optional
            BER threshold below which the instrument considers the pattern **Synchronized**. 
            Range: 1e-8 to 1e-1.
        sync : :obj:`bool`, optional
            If True, executes the **Pattern Synchronization** process. The PED shifts its internal 
            reference pattern to match the incoming bit stream.
        run : :obj:`bool`, optional
            If True, enables the error counting gate (Start Measurement). Requires successful synchronization.
        stop : :obj:`bool`, optional
            If True, disables the error counting gate (Stop Measurement).
        CHs : :obj:`int` or :obj:`list[int]`, optional
            Specific channels to configure. If None, applies to all channels.

        Notes
        -----
        **Difference between Alignment and Synchronization:**
        
        To measure BER correctly, the PED must perform two distinct operations:

        1.  **Alignment (Physical Layer - ``center_delay``, ``center_offset``)**: 

            The instrument adjusts the sampling point (time and voltage) to position it in the center 
            of the "Eye Diagram". 

            *   Without alignment, the PED samples noise or signal edges, resulting in a BER ~0.5.
            *   This process is analog and optimizes the signal quality reading.

        2.  **Synchronization (Logical Layer - ``sync``)**: 

            The instrument shifts the bits of its internal reference pattern to match the sequence of the incoming data stream.

            *   This happens **after** alignment. If the signal is not aligned (sampling noise), synchronization will fail.
            *   Successful sync forces the BER to 0 (or very low values) assuming the link is healthy.

        **Execution Order:**

        When multiple parameters are passed, this method executes them in the following order to ensure stability:

        1. Pattern Configuration (Type, PRBS/Data)
        2. Alignment Thresholds (Eye Threshold)
        3. Physical Alignment (Center Delay -> Center Offset -> Manual Values)
        4. Synchronization Configuration (Sync Threshold, Type)
        5. Execution of Synchronization (Sync)
        6. Gating (Run/Stop)

        Examples
        --------
        Full setup sequence: Configure PRBS31, align the eye, sync the pattern, and start measuring.

        >>> ped = PED4002('USB0::...')
        >>> ped(patt_type='PRBS', 
        ...     prbs=31, 
        ...     center_delay=True,   # Find horizontal eye center
        ...     center_offset=True,  # Find vertical eye center
        ...     sync=True,           # Lock pattern
        ...     run=True)            # Start counting errors
        """
        CHs = self._check_channels(CHs)

        # 1. Pattern Configuration
        if patt_type is not None:
            self.patt_type(patt_type, CHs)
        if patt_len is not None:
            self.patt_len(patt_len, CHs)
        if prbs is not None:
            self.prbs(prbs, CHs)
        if data is not None:
            self.data(data, start_addr=1, CHs=CHs)

        # 2. Alignment / Physical Layer
        if eye_threshold is not None:
            self.eye_threshold(eye_threshold, CHs)
        
        # Auto-alignment processes take time and change delay/offset values
        if center_delay:
            self.center_delay(CHs, wait=True)
        if center_offset:
            self.center_offset(CHs, wait=True)
        
        # Manual overrides (applied after auto-align if specified)
        if offset_mV is not None:
            self.offset(offset_mV, CHs)
        if delay_ps is not None:
            self.delay(delay_ps, CHs)

        # 3. Synchronization / Logical Layer
        if sync_threshold is not None:
            self.sync_threshold(sync_threshold, CHs)        
        if sync:
            # Sync requires a good physical signal (alignment) first
            self.sync(CHs, wait=True)

        # 4. Measurement Gating
        if run:
            self.run(CHs)
        if stop:
            self.stop(CHs)
            
        return self
    
    def __call__(self, *args, **kwargs):
        return self.setup(*args, **kwargs)

    # ==============================================================
    # Status summary
    # ==============================================================

    def get_metadata(self, ch : int=1):
        """Retrieve a summary of the current PED configuration for the specified channel as a dictionary."""
        metadata = {
            'PATT_TYPE': self.get_patt_type(ch)[0],
            'PATT_LEN': self.get_patt_len(ch)[0],
            'PRBS': self.get_prbs(ch)[0],
            'EYE_THR': self.get_eye_threshold(ch)[0],
            'OFFSET': self.get_offset(ch)[0],
            'V_EDGES': self.get_voltage_edges(ch)[0],
            'DELAY': self.get_delay(ch)[0],
            'T_EDGES': self.get_time_edges(ch)[0],
            'SYNC_THR': self.get_sync_threshold(ch)[0],
            'IS_SYNC': self.is_sync(ch)[0],
            'IS_RUNNIG': self.is_running(ch)[0],
            'BER': self.get_ber(ch)[0],
            'ERR_COUNT': self.get_error_count(ch)[0],
            'BIT_COUNT': self.get_bit_count(ch)[0],
            'FREQUENCY': self.get_frequency(ch)[0],
        }
        return metadata

    def print_setup(self, ch : int=1):
        """Print a summary of the current PED configuration for the specified channel."""

        metadata = self.get_metadata(ch)

        print(f"=========== SETUP CH{ch} ===========")
        for key, value in metadata.items():
            print(f"{1*' ' + key + (10 - len(key))*' '}: {value}")
        print("=================================")









import serial
import socket
class IDPhotonics:
    """
    Minimal SCPI driver for IDPhotonics lasers

    .. rubric:: Attributes
    .. autosummary::

        ~IDPhotonics.usb
        ~IDPhotonics.host
        ~IDPhotonics.port
        ~IDPhotonics.serial
        ~IDPhotonics.socket

    .. rubric:: Main methods
    .. autosummary::

        __init__
        wavelength
        get_wavelength
        power
        get_power
        fine_tune
        output
        close
    """
    usb = False
    """Use USB connection (True) or Ethernet (False)."""
    host = '192.168.0.1'
    """IP address of the device."""
    port = 2000
    """Port for socket or USB connection. If usb=True, this is the COM port number."""
    serial = None
    """PySerial object instance (if usb=True)."""
    socket = None
    """Socket object instance (if usb=False)."""

    def __init__(self, host='192.168.0.1', port=2000, timeout=0, usb=False):
        self.usb = usb
        self.host = host
        self.port = port

        if self.usb:
            self.serial = serial.Serial(self.port, 115200, timeout=timeout)
        else:
            self.socket = socket.socket()
            if timeout == 0:
                self.socket.settimeout(None)
            else:
                self.socket.settimeout(timeout)
            self.socket.connect((self.host, int(self.port)))

    def _send(self, command:str, verbose:int=0) -> str:
        '''
        sends a scpi command to device
        command: SCPI command string, refer to documentation for allowed commands and syntax
        verbose: verbosity setting. 1 for RX, 2 for RX and TX, no printing for any other setting
        '''
        command = command.rstrip('\n')
        if verbose>=2:
            print('TX: ' + command)
        if self.usb:
            self.serial.write((command + '\n').encode())
            self.serial.flush()
            reply = ''
            while reply.find('\n') < 0:
                reply = reply + self.serial.read(255).decode('latin1')
        else:
            self.socket.sendall(bytearray(command + '\n', 'utf-8'))
            reply = ''
            while reply.find('\n') < 0:
                reply = reply + self.socket.recv(1024).decode('utf-8')

        if verbose>=2:
            print('RX: ' + reply)
        elif verbose == 1:
            print(reply)
        return reply
    
    def close(self):
        '''
        Close connection
        '''
        if not self.usb:
            self.socket.close()
        else:
            self.serial.close()
        print("IDPhotonics: disconnected")

    def get_wavelength(self, ch=1) -> float:
        '''
        returns the current wavelength of the laser in nm, at the specified channel
        '''
        return float(self._send(f'WAV? 1,1,{ch}').strip(';\r\n'))
    
    def wavelength(self, wavelength:float, ch=1):
        '''
        sets the wavelength of the laser in nm, at the specified channel
        '''
        limits = np.array(self._send(f'wav:lim? 1,1,{ch}').strip(',;\r\n').split(','), dtype=float)
        if wavelength < limits[0] or wavelength > limits[1]:
            raise ValueError(f'Wavelength {wavelength} out of range. Must be between {limits[0]} and {limits[1]} nm')

        self._send(f'WAV 1,1,{ch},{wavelength}')
        self._send(f'bwai 1,1,{ch}')
        return self

    def get_power(self, ch=1) -> float:
        '''
        returns the current power of the laser in dBm, at the specified channel
        '''
        return float(self._send(f'POW? 1,1,{ch}').strip(';\r\n'))
    
    def power(self, power:float, ch=1):
        '''
        sets the power of the laser in dBm, at the specified channel
        '''
        limits = np.array(self._send(f'lim? 1,1,{ch}').strip(';\r\n').split(','), dtype=float)[-2:]
        if power < limits[0] or power > limits[1]:
            raise ValueError(f'Power {power} out of range. Must be between {limits[0]} and {limits[1]} dBm')

        self._send(f'POW 1,1,{ch},{power}')
        self._send(f'bwai 1,1,{ch}')
        return self

    def fine_tune(self, offset, ch=1):
        '''
        fine tunes the laser frequency in GHz, at the specified channel
        '''
        limit = float(self._send(f'Offset:LIMit? 1,1,{ch}').strip(';\r\n'))
        if np.abs(offset) > limit:
            raise ValueError(f'Offset out of range. Must be between {-limit} and {limit}')
        
        self._send(f'Offset 1,1,{ch},{offset}')
        self._send(f'bwai 1,1,{ch}')
        return self

    def output(self, value:bool, ch=1):
        '''
        Enables or disables the output of the laser at the specified channel. To enable all lasers use `ch='*'`. This method wait until output power is stable. 
        '''
        self._send(f'State 1,1,{ch},{value}')
        self._send(f'bwai 1,1,{ch}')

        if ch != '*':
            return bool(int(self._send(f'State? 1,1,{ch}').strip(';\r\n'))) == value
        else: 
            outputs = np.array(self._send(f'State? 1,1,*').strip(';\r\n').replace('\n', ',').split(',')[3::4], dtype=int)
            if value == 1: 
                return outputs.prod() == 1
            else:
                return outputs.sum() == 0
            
    def __call__(self,
                wavelength: float = None,
                power: float = None,
                output: bool = None,
                ch: int = 1
                ):
        '''
        Convenience method to set wavelength and power in a single call.
        '''
        if wavelength is not None:
            self.wavelength(wavelength, ch)
        if power is not None:
            self.power(power, ch)
        if output is not None:
            self.output(output, ch)
        return self
    
    def get_metadata(self, ch: int=1):
        """Retrieve a summary of the current laser configuration for the specified channel as a dictionary."""
        metadata = {
            'WAVELENGTH_NM': self.get_wavelength(ch),
            'POWER_DBM': self.get_power(ch),
        }
        return metadata
    
    def print_setup(self, ch: int=1):
        """Print a summary of the current laser configuration for the specified channel."""

        metadata = self.get_metadata(ch)

        print(f"=========== SETUP CH{ch} ===========")
        for key, value in metadata.items():
            print(f"{1*' ' + key + (15 - len(key))*' '}: {value}")
        print("=================================")




class LeCroy_WavExp100H:
    """
    LeCroy Wave Expert 100H - minimal, extensible VISA wrapper for Teledyne LeCroy MAUI/XStreamDSO scopes adquisition.

    .. rubric:: Attributes
    .. autosummary::

        ~PED4002.inst

    .. rubric:: Main methods
    .. autosummary::

        __init__
        run
        stop
        single
        autoset
        acquire_waveform
        close 

    Basic workflow example
    ----------------------
    >>> scope = LeCroy_WavExp100H(addr_ID)
    >>> t, v = scope.acquire_waveform(ch=1)  
    >>> scope.close()
    """

    def __init__(self, addr_ID: str = None, timeout_ms: int = 10000):
        if addr_ID: 
            self.inst = visa.ResourceManager().open_resource(addr_ID)
            self.inst.timeout = 10000 # timeout in milliseconds
            self.inst.chunk_size = 2**22
            self.inst.endian = 'little'
            print(self._query('*IDN?'))

    def __del__(self):
        try:
            self.inst.close()
        except AttributeError:
            pass
        except Exception as e:
            raise e

    # -------------------------
    # Low-level wrappers
    # -------------------------

    def _query(self, cmd: str) -> str:
        try:
            resp = self.inst.query(cmd).strip()
            return resp
        except AttributeError:
            print(cmd)
            return ''
        except Exception as e:
            raise e
        
    def _write(self, cmd: str) -> None:
        try:
            self.inst.write(cmd)
        except AttributeError:
            print(cmd)
        except Exception as e:
            raise e
        
    def _wait_until_idle(self, timeout_s: int = 5) -> None:
        """Ask the instrument to wait until it is idle.
        """
        _ = self._query(rf"""vbs? 'return=app.WaitUntilIdle({timeout_s})' """)

    
    # ------------------------------------------------------
    # Configuration and Acquisition
    # ------------------------------------------------------

    def stop(self):
        """Stop any ongoing acquisition."""
        self._write(r"""vbs 'app.acquisition.triggermode="Stopped"' """)
        self._wait_until_idle()

    def run(self):
        """Run acquisition again"""
        self._write(r"""vbs 'app.acquisition.triggermode="Normal"'""")
        self._wait_until_idle()
    
    def single(self):
        """Arm the scope for a single acquisition."""
        self._write(r"""vbs 'app.acquisition.triggermode="Single"' """)
        self._wait_until_idle()

    def autoset(self):
        """Run AutoSetup on the oscilloscope (convenience wrapper)."""
        self._write(r"""vbs 'app.AutoSetup' """)
        self._wait_until_idle()

    def _get_wavedesc(self, ch):
        """
        Parses the output string from scope.query('C#:INSPECT? WAVEDESC') into a dictionary.

        The string contains lines separated by '\r\n', each with the format 'KEY : VALUE'.

        Parameters
        ----------
        ch : int
            LeCroy channel from which to obtain the description. 

        Returns
        -------
        dict
            Dictionary with parsed keys and values from the waveform descriptor.
        """
        desc = self._query(f'C{ch}:INSPECT? WAVEDESC')
        lines = desc.strip().split('\r\n')
        metadata = {}
        for line in lines:
            if ' : ' in line:
                key, value = line.split(' : ', 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value
        return metadata
        
    # ------------------------------------------------------
    # Waveform readout and parsing
    # ------------------------------------------------------

    def _parse_IEEE488p2_block(self, raw: bytes, dtype=np.int8) -> np.ndarray:
        """Parse a LeCroy binary waveform block (#nXXXXXXXXX<data>)."""
        i = raw.find(b'#')
        if i < 0:
            raise ValueError("No block header found")
        n_digits = int(chr(raw[i+1]))
        length = int(raw[i+2:i+2+n_digits].decode())
        start = i + 2 + n_digits
        data = raw[start:start+length]
        return np.frombuffer(data, dtype=dtype)
    
    def _extract_value(self, desc: str, key: str):
        """Extract a numeric or string value from the waveform descriptor."""
        pattern = rf"{key}\s*[:=]\s*(.+)"
        match = re.search(pattern, desc)
        if not match:
            raise KeyError(f"Key '{key}' not found in descriptor")

        raw_value = match.group(1).strip()

        try:
            num = float(raw_value)
            return num
        except ValueError:
            return raw_value


    def acquire_waveform(self, ch: int = 1, points=None):
        """Acquire waveform data from the specified channel.

        Parameters
        ----------
        ch : int
            Channel number to acquire from (1-4).
        points : int, optional
            Number of points to acquire. If None, acquires all available points.
        Returns
        -------
        t : np.ndarray
            Time array corresponding to the waveform samples.
        v : np.ndarray
            Voltage array of the acquired waveform.
        """
        # ---- Set number of points to acquire ----
        self._write(f'WFSU SP,0,NP,{points if points else 0},FP,0,SN,0') 

        # ---- Read waveform data ----
        self._write(f'C{ch}:WF? DAT1')
        raw_bytes = self.inst.read_raw()
        data = self._parse_IEEE488p2_block(raw_bytes)

        # ---- Data scaling ----
        # desc = self._query(f'C{ch}:INSPECT? WAVEDESC')
        desc = self._get_wavedesc(ch)

        VERT_GAIN = float(desc.get("VERTICAL_GAIN", 0))
        VERT_OFFSET = float(desc.get("VERTICAL_OFFSET", 0))
        HORIZ_INTERVAL = float(desc.get("HORIZ_INTERVAL", 0))
        HORIZ_OFFSET = float(desc.get("HORIZ_OFFSET", 0))

        v = data * VERT_GAIN - VERT_OFFSET
        t = np.arange(len(v)) * HORIZ_INTERVAL + HORIZ_OFFSET
        return t, v
    
    def close(self):
        """Close the connection to the instrument."""
        self.__del__()
        print("LeCroy: disconnected")


# import struct
# from smbus2 import SMBus, i2c_msg

# class TDCMX:
#     # Default I2C device address
#     DEFAULT_ADDRESS = 0x60
    
#     # Basic Operation commands
#     CMD_GET_STATUS = 0x00
#     CMD_RESET = 0x28
#     CMD_SET_FREQUENCY = 0x2E
#     CMD_GET_FREQUENCY = 0x2F
#     CMD_SET_DISPERSION = 0x30
#     CMD_GET_DISPERSION = 0x31
#     CMD_ENABLE_DEVICE = 0x1E
#     CMD_DISABLE_DEVICE = 0x1F

#     # Nominal settings commands
#     CMD_SET_STRTUP_BYTE = 0x34
#     CMD_GET_STRTUP_BYTE = 0x35
#     CMD_SET_NOMINAL_SETTINGS = 0x36
#     CMD_GET_NOMINAL_SETTINGS = 0x37

#     # Tunning information commands
#     CMD_GET_CHANNEL_PLAN = 0x3B

#     # General Device Information commands
#     CMD_GET_VERSION = 0x0F
#     CMD_READ_MANUFACTURER_NAME = 0x0E
#     CMD_READ_MODEL_NUMBER = 0x27
#     CMD_READ_SERIAL_NUMBER = 0x29
#     CMD_READ_MANUFACTURER_DATE = 0x2B

#     # Communication commands
#     CMD_SET_I2C_ADDRESS = 0x42

#     def __init__(self, bus_id=1, address=DEFAULT_ADDRESS):
#         """
#         Inicializa el dispositivo.
#         :param bus_id: ID del bus I2C (usualmente 1 en Raspberry Pi).
#         :param address: Dirección I2C del dispositivo (0x60 por defecto).
#         """
#         self.bus = SMBus(bus_id)
#         self.address = address

#     @staticmethod
#     def bytes2str(info_bytes: list[bytes]) -> str:
#         """Convierte bytes de info del dispositivo a string ASCII."""
#         info_str = bytes(info_bytes).split(b'\x00')[0]
#         return info_str.decode('ascii')
        
#     @staticmethod
#     def bytes2float(data_bytes: bytes | list[bytes]) -> float:
#         """Convierte 4 bytes Big-Endian a float32."""
#         return struct.unpack('>f', bytes(data_bytes))[0]
    
#     @staticmethod
#     def float2bytes(value: float) -> list[int]:
#         """Convierte un float32 a 4 bytes Big-Endian."""
#         return list(struct.pack('>f', value))
    
#     @staticmethod
#     def bytes2int(data_bytes: bytes | list[bytes]) -> int:
#         """Convierte 4 bytes Big-Endian a uint32."""
#         return struct.unpack('>I', bytes(data_bytes))[0]
    
#     @staticmethod
#     def byte2int8(byte: bytes) -> int:
#         """Convierte 1 byte Big-Endian a uint8."""
#         return struct.unpack('>B', byte)[0]

#     def _parse_status(self, status_bytes: bytes | list[bytes]) -> dict:
#         """
#         Decodifica los 4 bytes de estatus y devuelve la info en formato de diccionario.
#         """
#         value = self.bytes2int(status_bytes)

#         status = {}
#         status['BUSY_ERROR'] = (value >> 23) & 1
#         status['OVERRUN_ERROR'] = (value >> 22) & 1
#         status['COMMAND_ERROR'] = (value >> 21) & 1
#         status['TDCMX_ACTIVE'] = (value >> 20) & 1
#         status['TDCMX_READY'] = (value >> 19) & 1
#         status['EEPROM_ERROR'] = (value >> 18) & 1

#         status['TEC4_LIMIT_REACHED'] = (value >> 7) & 1
#         status['TEC3_LIMIT_REACHED'] = (value >> 6) & 1
#         status['TEC2_LIMIT_REACHED'] = (value >> 5) & 1
#         status['TEC1_LIMIT_REACHED'] = (value >> 4) & 1

#         status['TEC4_IN_RANGE'] = (value >> 3) & 1
#         status['TEC3_IN_RANGE'] = (value >> 2) & 1
#         status['TEC2_IN_RANGE'] = (value >> 1) & 1
#         status['TEC1_IN_RANGE'] = value & 1

#         return status
          
#     def _send_command(self, cmd_id: int, data: list[int]=[]):
#         """Envía el comando y datos al dispositivo (Write Transaction)."""
#         try:
#             msg = i2c_msg.write(self.address, [cmd_id] + data)
#             self.bus.i2c_rdwr(msg)
#         except Exception as e:
#             print(f"Error escribiendo comando I2C: {e}")
#             raise

#     def _read_response(self, n_bytes: int=0):
#         """
#         Implementa el protocolo de Polling descrito en la Página 12 (Figura 7).
#         Lee repetidamente los 4 bytes de estatus hasta que el bit 'Busy' se limpie.
#         Luego lee la respuesta completa.
#         """
#         max_retries = 100
        
#         for _ in range(max_retries):
#             # Leer solo los 4 bytes de estatus
#             msg_status = i2c_msg.read(self.address, length=4)
#             self.bus.i2c_rdwr(msg_status)
#             status_bytes = list(msg_status)
            
#             status = self._parse_status(status_bytes)
            
#             if status['COMMAND_ERROR']:
#                 raise RuntimeError("El dispositivo TDCMX reportó un Command Error.")
            
#             if status['OVERRUN_ERROR']:
#                 raise RuntimeError("El dispositivo TDCMX reportó un Overrun Error.")
            
#             if status['EEPROM_ERROR']:
#                 raise RuntimeError("El dispositivo TDCMX reportó un EEPROM Error.")

#             if not status['BUSY_ERROR']:
#                 # Si no hay datos extra que leer (solo queríamos saber que terminó)
#                 if n_bytes == 0:
#                     return status
                
#                 # Si hay datos, hacemos la lectura final completa (Status 4 bytes + Data N bytes)
#                 # Página 12: "reading the full device answer (4 status bytes + N data bytes)"
#                 total_length = 4 + n_bytes
#                 msg_full = i2c_msg.read(self.address, total_length)
#                 self.bus.i2c_rdwr(msg_full)
#                 response = list(msg_full)

#                 status = self._parse_status(response[:4])
                
#                 # Retornamos el estatus y la parte de datos (bytes 4 en adelante)
#                 return status, response[4:]
            
#             # Esperar un poco antes de volver a consultar (10ms es el tiempo típico de proceso según PDF)
#             time.sleep(0.01)
            
#         raise TimeoutError("El dispositivo TDCMX permaneció ocupado (Busy) demasiado tiempo.")
    
#     def _query(self, cmd_id: int, data: list[int]=[], n_bytes: int=0):
#         """Envía un comando y lee la respuesta."""
#         self._send_command(cmd_id, data)
#         return self._read_response(n_bytes)

#     # ------------------------------------------------------
#     # Comandos principales

#     def get_status(self):
#         """Obtiene el estatus actual del dispositivo."""
#         status = self._query(self.CMD_GET_STATUS)
#         return self._parse_status(status[:4])
    
#     def reset(self):
#         """Reinicio por software"""
#         self._send_command(self.CMD_RESET)
#         time.sleep(0.3) # Esperar 300ms según PDF

#     def set_frequency(self, frequency_ghz):
#         """
#         Ajusta el Set Point de Frecuencia.
#         """
#         status, freq = self._query(self.CMD_SET_FREQUENCY, self.float2bytes(frequency_ghz), n_bytes=4) 
#         print('Frequency set response:', self.bytes2float(freq))
#         return status

#     def get_frequency(self):
#         """Lee la frecuencia actual."""
#         _, freq = self._query(self.CMD_GET_FREQUENCY, n_bytes=4)
#         return self.bytes2float(freq)

#     def set_dispersion(self, dispersion_ps_nm):
#         """Ajusta la dispersión."""
#         status, disp = self._query(self.CMD_SET_DISPERSION, self.float2bytes(dispersion_ps_nm), n_bytes=4)
#         print('Dispersion set response:', self.bytes2float(disp))
#         return status

#     def enable_device(self):
#         """Activa el control TEC."""
#         return self._query(self.CMD_ENABLE_DEVICE)

#     def disable_device(self):
#         """Desactiva el control TEC (Cmd 0x1F)."""
#         return self._query(self.CMD_DISABLE_DEVICE)

#     def get_device_info(self):
#         """Lee el número de serie (Cmd 0x29). Devuelve string."""
#         _, version = self._send_command(self.CMD_GET_VERSION, n_bytes=2)
#         _, manufac_name = self._send_command(self.CMD_READ_MANUFACTURER_NAME, n_bytes=256)
#         _, model_number = self._send_command(self.CMD_READ_MODEL_NUMBER, n_bytes=256)
#         _, serial_number = self._send_command(self.CMD_READ_SERIAL_NUMBER, n_bytes=256)
#         _, manufac_date = self._send_command(self.CMD_READ_MANUFACTURER_DATE, n_bytes=256)

#         return ', '. join([
#             f"v{self.byte2int8(version[0])}.{self.byte2int8(version[1])}",
#             self.bytes2str(manufac_name),
#             self.bytes2str(model_number),
#             self.bytes2str(serial_number),
#             self.bytes2str(manufac_date)
#         ])
        

        

