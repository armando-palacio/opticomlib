"""
.. rubric:: Functions
.. autosummary::

   search_inst
   SYNC                  
   GET_EYE_v2            

.. rubric:: Classes
.. autosummary::

   PPG3204         
"""
import numpy as np
import scipy.signal as sg

from .typing import binary_sequence, electrical_signal, eye, Array_Like, Number
from typing import Literal, Union
from .utils import tic, toc, str2array, nearest

import pyvisa as visa
import warnings
import time


def search_inst():
    """**Intruments search**
    
    Search for the available instruments in the system and print the IDs.
    """
    rm = visa.ResourceManager()
    print(rm.list_resources())


def SYNC(signal_rx: electrical_signal, 
         slots_tx: binary_sequence, 
         sps: int):
    r"""**Signal Synchronizer**

    Synchronizes the received signal with the transmitted signal to determine the starting position in the received signal for further processing. 
    This is done by performing a correlation between the received signal and the transmitted signal and finding the maximum correlation position
    and shifting the received signal to that position (deleting the samples before the maximum correlation position).

    Parameters
    ----------
    signal_rx : :obj:`electrical_signal`
        The received digital signal (from the oscilloscope or an ADC).
    slots_tx : :obj:`binary_sequence`
        The transmitted slots sequence.
    sps : :obj:`int`
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
    if not isinstance(sps, int):
        raise TypeError('The "sps" must be an integer to perform synchronization.')

    signal_tx = np.kron(slots_tx.data, np.ones(sps))
    signal_rx = signal_rx.signal

    if len(signal_rx)<len(signal_tx): 
        raise BufferError('The length of the received vector must be greater than the transmitted vector!!')

    l = len(signal_tx)
    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[l::-1], mode='valid') # Correlation of the transmitted signal with the received signal in a window of 2*l (sufficient to find a maximum)

    if np.max(corr) < 3*np.std(corr): 
        raise ValueError('No correlation maximum found!!') # false positive
    
    i = np.argmax(corr)

    signal_sync = electrical_signal(signal_rx[i:-(l-i)])
    signal_sync.execution_time = toc()
    return signal_sync, i


def GET_EYE_v2(sync_signal: electrical_signal, slots_tx: binary_sequence, sps:int, nslots:int=8192):
    r"""**Eye diagram parameters v2**

    Estimate the means and standard deviations of levels 0 and 1 in the ``sync_signal`` 
    knowing the transmitted sequence ``slots_tx``. It separates the received signal levels
    corresponding to transmitted level 0 and 1 and estimates the means and standard deviations. 

    Parameters
    ----------
    sync_signal : electrical_signal
        Synchronized digital signal in time with the transmitted signal.
    slots_tx : binary_sequence
        Transmitted bit sequence.
    sps : int
        Number of samples per slot of the digitalized signal ``sync_signal``.
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
    sps = sync_signal.sps()

    eye_dict = {}

    eye_dict['sps'] = sps


    rx = sync_signal[:nslots*sps].signal + sync_signal[:nslots*sps].noise; eye_dict['y'] = rx
    tx = np.kron(slots_tx.data[:nslots], np.ones(sps))

    unos = rx[tx==1]; eye_dict['unos']=unos
    zeros = rx[tx==0]; eye_dict['zeros']=zeros

    t0 = np.kron(np.ones(zeros.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t0']=t0
    t1 = np.kron(np.ones(unos.size//sps), np.linspace(-0.5, 0.5, sps, endpoint=False)); eye_dict['t1']=t1

    eye_dict['i']=sps//2

    unos_ = unos[(t1<0.05) & (t1>-0.05)]
    zeros_ = zeros[(t0<0.05) & (t0>-0.05)]

    mu0 = np.mean(zeros_).real; eye_dict['mu0'] = mu0
    mu1 = np.mean(unos_).real; eye_dict['mu1'] = mu1

    s0 = np.std(zeros_).real; eye_dict['s0'] = s0
    s1 = np.std(unos_).real; eye_dict['s1'] = s1

    return eye(eye_dict)


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
        reset
        set_patt_len
        get_patt_len
        set_mode
        get_mode
        set_prbs_order
        get_prbs_order
        set_data
        get_data
        set_bits_shift
        get_bits_shift
        enable_outputs
        disable_outputs
        set_freq
        get_freq
        set_skew
        get_skew
        set_output_voltage
        get_output_voltage
        set_offset
        get_offset
        __call__
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
    """The order of polynomial generator for PRBS mode, [7,9,11,15,23,31]"""
    MAX_MEMORY_LEN = 2**21
    """Maximum length of the memory of the PPG3204, 2^21 = 2097152 (2M) for each channel."""
    MAX_CHUNK_LEN = 1024
    """Maximum length of the data to send in a single command, 1024 bits."""
    MIN_SKEW = -25e-12
    """Minimum skew, -25 ps"""
    MAX_SKEW = 25e-12
    """Maximum skew, 25 ps"""

    def __init__(self, addr_ID: str = None):
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
            """A connection (session) to the PPG."""
            self.inst.timeout = 10000 # timeout in milliseconds
            print(self._query('*IDN?'))
        
    def __del__(self):
        try:
            self.inst.clear()
            self.inst.close()
        except AttributeError:
            pass
        except Exception as e:
            print(e)

    def _query(self, command: str):
        """Query the PPG."""
        try:
            resp = self.inst.query(command)
            if resp == '\n\n':
                raise EOFError(f'Invalid command {command}')  # invalid command
            if resp == '\n':
                return True  # when write commands are executed
            return resp  # when query commands are executed
        except AttributeError:
            print(command)
            return 0
        except Exception as e:
            raise e
    
    def _check_channels(self, channels):
        """Check if channels are in the correct format and return it as array."""
        if channels is not None and not isinstance(channels, (int,) + Array_Like):
            raise ValueError('`channels` is not in the correct format')
        
        if channels is not None:
            if isinstance(channels, int):
                channels = np.array([channels], dtype=int)
            else:
                channels = np.array(channels, dtype=int)

            if (channels < 1).any() or (channels > self.CHANNELS).any() or channels.size > self.CHANNELS:
                msg = 'The channels number is out of the range of the PPG3204. Setting to the limits.'
                warnings.warn(msg)
                channels = channels.clip(1, self.CHANNELS)[:self.CHANNELS]
        else:
            channels = np.arange(1, self.CHANNELS+1)
        return channels


    def reset(self):
        """Reset the PPG to its default state."""
        self._query('*RST')


    def set_patt_len(self, patt_len: Union[int, list[int]], CHs: Union[int, list[int]] = None):
        """Set the length of the pattern

        Fix a pattern length for each channel passed as argument. 
        
        - Range: 2 to 2097152 bits (2 Mbits/channel). 
        - Resolution: 1 bit.
        
        Parameters
        ----------
        patt_len : :obj:`int` or :obj:`Array_Like(int)`
            Pattern length for every channel specified in ``CHs``.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to set the pattern length.

        Raises
        ------
        ValueError
            If ``patt_len`` is not in the correct format or type.
        
        Warn
        ----
        UserWarning
            If the pattern length is out of range.
        """
        if not isinstance(patt_len, (int,) + Array_Like):
            raise ValueError('`patt_len` is not in the correct format')
        
        CHs = self._check_channels(CHs)

        if isinstance(patt_len, int):
            patt_len = np.tile([patt_len], CHs.size)
        else:
            patt_len = np.array(patt_len)
        
        if (patt_len < self.PATT_LEN_MIN).any() or (patt_len > self.PATT_LEN_MAX).any():
            msg = f'The pattern length is out of the range of the PPG3204. Setting to the limits.'
            warnings.warn(msg)
            patt_len = patt_len.clip(self.PATT_LEN_MIN, self.PATT_LEN_MAX) 
        
        for ch, pl in zip(CHs, patt_len):
            self._query(f':DIG{ch}:PATT:LENG {pl}')


    def get_patt_len(self, CHs: Union[int, list[int]]=None):
        """Get the current length of pattern for specified channels
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the pattern length. 
        
        Returns
        -------
        patt_len : :obj:`np.ndarray`
            Every channel pattern length.
        """
        CHs = self._check_channels(CHs)
        return np.array([int(self._query(f':DIG{ch}:PATT:LENG?')) for ch in CHs])


    def set_mode(self, mode: Literal['data', 'prbs'], CHs: Union[int, list[int]] = None):
        """Set mode of the PPG3204 for each channels specified.

        Parameters
        ----------
        mode : :obj:`str` {``'DATA'``, ``'PRBS'``} 
            Work mode of the PPG.    
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the mode.
        """
        CHs = self._check_channels(CHs)
        
        if mode.upper() not in ['DATA', 'PRBS']:
            raise ValueError('`mode` must be "data" or "prbs"')
        
        mode = np.tile([mode.upper()], CHs.size)
        
        for ch, t in zip(CHs, mode):
            self._query(f':DIG{ch}:PATT:TYPE {t}')


    def get_mode(self, CHs: Union[int, list[int]] = None):
        """Get mode of the PPG3204 for each channels specified, can be 'DATA' or 'PRBS'

        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the mode.

        Returns
        -------
        mode : :obj:`np.ndarray`
            Every channel mode.
        """
        CHs = self._check_channels(CHs)
        return np.array([self._query(f':DIG{ch}:PATT:TYPE?') for ch in CHs])
    

    def set_prbs_order(self, order: Union[int, list[int]], CHs: Union[int, list[int]] = None):
        """Set the order of polynomial generator for PRBS mode.
        
        Parameters
        ----------
        order : :obj:`int` or :obj:`Array_Like(int)`
            order of the polynomial generator.
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
        
        if not isinstance(order, (int,) + Array_Like):
            raise ValueError('`order` is not in the correct format')
        
        if isinstance(order, int):
            order = np.tile([order], CHs.size)
        else:
            order = np.array(order)
        
        for ch, ord in zip(CHs, order):
            if ord not in self.PRBS_ORDERS:
                old_ord = ord
                ord = nearest(self.PRBS_ORDERS, int(ord)) 
                msg = f'PRBS order {old_ord} in CH:{ch} is not correct, it will be set to nearest value {ord}'
                warnings.warn(msg)
            self._query(f':DIG{ch}:PATT:PLEN {ord}')

    
    def get_prbs_order(self, CHs: Union[int, list[int]] = None):
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
    

    def set_data(self, data: Union[str, np.ndarray], start_addrs: int=1, CHs: Union[int, list[int]] = None):
        """Set the data of the pattern.

        Programs the pattern data memory. Each byte of pattern data is a character (0 or 1)
        representing one bit of pattern data. The start address can be any bit location, MAX_MEMORY_LEN - 1.
        MAX_MEMORY_LEN is :math:`2^{21} = 2097152` (2M) for each channel. 
        
        Parameters
        ----------
        data : :obj:`str` or :obj:`Array_Like(int)`
            Data to set to the specified channels (use :obj:`str` type only when ``CHs`` is :obj:`int` or ``None``).
        start_addrs : :obj:`int`, optional
            Start address of the data to set in the pattern memory. The range is from 1 to 2^21. Default is 1.
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
        
        if not isinstance(data, (str,) + Array_Like):
            raise ValueError('`data` is not in the correct format')
        
        if len(data) > self.MAX_MEMORY_LEN-start_addrs+1:
            msg = 'The length of the data is greater than the maximum memory length minus the start address. Setting to the nearest value.'
            warnings.warn(msg)
            data = data[:self.MAX_MEMORY_LEN-start_addrs+1]

        if isinstance(data, str):
            data = str2array(data, bool).astype(np.uint8)
        else:
            data = np.array(data, dtype=bool).astype(np.uint8)
        
        if data.ndim == 1:
            data = np.tile(data, (CHs.size, 1))
        
        for ch, data_ch_i in zip(CHs, data):

            if data_ch_i.size > self.MAX_CHUNK_LEN:
                chunks = np.split(data_ch_i, self.MAX_CHUNK_LEN*np.arange(1, data_ch_i.size//self.MAX_CHUNK_LEN + 1))
            else:
                chunks = [data_ch_i]
            
            addr = start_addrs
            for chunk in chunks:
                p = addr # memory position
                n = chunk.size # data length
                k = len(str(n)) # n digits number
                data_ = ''.join(chunk.astype(str)) # binary data string
                self._query(f':DIG{ch}:PATT:DATA {p},{n},#{k}{n}{data_}')
                addr += n


    def get_data(self, size: int, start_addrs: int=1, CHs: Union[int, list[int]] = None):
        """Get the data of the pattern for each specified channel 
        
        Parameters
        ----------
        size : :obj:`int`
            Size of the data to get from the pattern memory.
        start_addrs : :obj:`int`, optional
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
            If ``start_addrs`` or ``size`` are not integers.
        """
        CHs = self._check_channels(CHs)

        if not isinstance(start_addrs, int):
            raise ValueError('`start_addrs` must be an integer')
        if not isinstance(size, int):
            raise ValueError('`size` must be an integer')
        
        if start_addrs < 1 or start_addrs > self.MAX_MEMORY_LEN:
            msg = f'`start_addrs` must been between 1 and {self.MAX_MEMORY_LEN}. Setting to the nearest value.'
            warnings.warn(msg)
            start_addrs = np.clip(start_addrs, 1, self.MAX_MEMORY_LEN)

        if size < 1 or size > self.MAX_MEMORY_LEN - start_addrs + 1:
            msg = f'`size` must been between 1 and (MAX_MEMORY_LEN - start_addrs+1)={self.MAX_MEMORY_LEN - start_addrs + 1}. Setting to the nearest value.'
            warnings.warn(msg)
            size = np.clip(size, 1, self.MAX_MEMORY_LEN - start_addrs + 1)

        if size > self.MAX_CHUNK_LEN:
            bits_count = np.concatenate((np.tile([self.MAX_CHUNK_LEN], size//self.MAX_CHUNK_LEN), [size%self.MAX_CHUNK_LEN]))
        else:
            bits_count = [size]

        data = []
        for ch in CHs:
            addr = start_addrs
            data_ch = []
            
            for bit_count in bits_count:
                b = self._query(f':DIG{ch}:PATT:DATA? {addr},{bit_count}')
                k = int(b[1])
                data_ch.append( str2array(b[k+2:-1], bool).astype(np.uint8) )
                addr += bit_count

            data.append(np.array(data_ch))
        return np.array(data)


    def set_bits_shift(self, bsh: Union[int, list[int]], CHs: Union[int, list[int]] = None):
        """Set the bits shift of the pattern
        
        Parameters
        ----------
        bsh : :obj:`int` or :obj:`Array_Like(int)`
            Bits shift to set to the specify channels
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
        
        if not isinstance(bsh, Number + Array_Like):
            raise ValueError('`bsh` is not in the correct format')
        
        if isinstance(bsh, Number):
            bsh = np.tile([bsh], CHs.size)
        else:
            bsh = np.array(bsh)
        
        for ch, b in zip(CHs, bsh):
            self._query(f':DIG{ch}:PATT:BSH {b}')


    def get_bits_shift(self, CHs: Union[int, list[int]] = None):
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


    def enable_outputs(self, CHs: Union[int, list[int]] = None):
        """Enable the output of the channels
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to enable the output. If ``CHs=None`` all channels will be enabled.
        """
        CHs = self._check_channels(CHs)   
        for ch in CHs:
            self._query(f':OUTP{ch} ON')


    def disable_outputs(self, CHs: Union[str, int, list[str], list[int]] = None):
        """Disable the output of the channels.
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to disable the output. If ``CHs=None`` all channels will be disabled.
        """
        CHs = self._check_channels(CHs)
        for ch in CHs:
            self._query(f':OUTP{ch} OFF')


    def set_freq(self, freq: float):
        """Set the bit rate of the pattern

        - *Range*: 1.5 GHz to 32 GHz
        - *Resolution*: 10 kb/s
        - *Accuracy*: :math:`\pm 5` ppm
        
        Parameters
        ----------
        freq : :obj:`float`
            Frequency of the pattern in Hz. The range is from 1.5 GHz to 32 GHz.

        Warns
        -----
        UserWarning
            If the frequency is out of the range of the PPG3204.
        """
        if freq < self.FREQ_MIN or freq > self.FREQ_MAX:
            msg = f'The frequency is out of the range of the PPG3204. Setting to the limits.'
            warnings.warn(msg)
            freq = np.clip(freq, self.FREQ_MIN, self.FREQ_MAX)

        self._query(f':FREQ {freq:.1e}')
    

    def get_freq(self):
        """Get the frequency of the pattern.
        
        Returns
        -------
        freq: :obj:`float`
            Bit Rate of the pattern in bits/s.
        """
        return float(self._query(':FREQ?'))
    

    def set_skew(self, skew: Union[float, list[float]], CHs: Union[int, list[int]] = None):
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

        if not isinstance(skew, Number + Array_Like):
            raise ValueError('`skew` is not in the correct format')
        
        if isinstance(skew, Number):
            skew = np.tile([skew], CHs.size)
        else:
            skew = np.array(skew)

        if (skew < self.MIN_SKEW).any() or (skew > self.MAX_SKEW).any():
            msg='The skew is out of the range of the PPG3204. Setting to the limits.'
            warnings.warn(msg)
            skew = skew.clip(self.MIN_SKEW, self.MAX_SKEW)
        
        for ch, s in zip(CHs, skew):
            self._query(f':SKEW{ch} {s}')


    def get_skew(self, CHs: Union[int, list[int]] = None):
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


    def set_output_voltage(self, amplitud: Union[float, list[float]], CHs: Union[int, list[int]] = None):
        """Set the peak-to-peak output voltage of each channel, in volts.
        
        Parameters
        ----------
        amplitud : :obj:`float` or :obj:`Array_Like`
            Amplitud to set to the specify channels
        CHs : :obj:`int` or :obj:`Array_Like`, optional
            Channels to set the amplitud. If ``CHs=None`` amplitud will be fixed in all channels.
        """
        CHs = self._check_channels(CHs)
        
        if not isinstance(amplitud, Number + Array_Like):
            raise ValueError('`amplitud` is not in the correct format')
        
        if isinstance(amplitud, Number):
            amplitud = np.tile([amplitud], CHs.size)
        else:
            amplitud = np.array(amplitud)

        if amplitud.any() < self.AMPLITUDE_MIN or amplitud.any() > self.AMPLITUDE_MAX:
            msg = 'The amplitude is out of the range of the PPG3204. Setting to the limits.'
            warnings.warn(msg)
            amplitud = amplitud.clip(self.AMPLITUDE_MIN, self.AMPLITUDE_MAX) 
        
        for ch, amp in zip(CHs, amplitud):
            self._query(f':VOLT{ch}:POS {amp:.1f}v')

    def get_output_voltage(self, CHs: Union[int, list[int]] = None):
        """Get the peak-to-peak output voltage of each channel, in volts.
        
        Parameters
        ----------
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            List of channels to get the amplitud.

        Returns
        -------
        Vout : :obj:`np.ndarray`
            Every channel output voltage.
        """
        CHs = self._check_channels(CHs)
        return np.array([float(self._query(f':VOLT{ch}:POS?')) for ch in CHs])


    def set_offset(self, offset: Union[float, list[float]], CHs: Union[int, list[int]] = None):
        """Set the offset of the channels
        
        Parameters
        ----------
        offset : :obj:`float` or :obj:`Array_Like(float)`
            Offset to set to the specify channels
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the offset. If ``CHs=None`` offset will be fixed in all channels.

        Raises
        ------
        ValueError
            If ``offset`` is not in the correct format.

        Warns
        -----
        UserWarning
            If the offset is out of the range of the PPG3204.

        Notes
        -----
        
        **Offset adjust** relative to nominal position. 
            - **Range**: -2 to 3 V
        """
        CHs = self._check_channels(CHs)

        if not isinstance(offset, Number + Array_Like):
            raise ValueError('`offset` is not in the correct format')
        
        if isinstance(offset, Number):
            offset = np.tile([offset], CHs.size)
        else: 
            offset = np.array(offset)
        
        if (offset < self.OFFSET_MIN).any() or (offset > self.OFFSET_MAX).any():
            msg = 'The offset is out of the range of the PPG3204. Setting to the limits.'
            warnings.warn(msg)
            offset = offset.clip(self.OFFSET_MIN, self.OFFSET_MAX)

        for ch, off in zip(CHs, offset):
            if off < 0:
                self._query(f':VOLT{ch}:NEG:OFFS {off:.1f}v')
            else:
                self._query(f':VOLT{ch}:POS:OFFS {off:.1f}v')


    def get_offset(self, CHs: Union[int, list[int]] = None):
        """Get the offset of the channels
        
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
        return np.array([float(self._query(f':VOLT{ch}:OFFS?')) for ch in CHs])


    def __call__(self, 
               freq: float = None, 
               patt_len: Union[int, list[int]] = None, 
               Vout: Union[float, list[float]] = None,
               offset: Union[float, list[float]] = None,
               bsh: Union[int, list[int]] = None, 
               skew: Union[float, list[float]] = None,
               mode: Literal['DATA', 'PRBS'] = None, 
               order: Union[int, list[int]] = None, 
               data: Union[np.ndarray, list[np.ndarray]] = None,
               CHs: Union[int, list[int]] = None):
        """ Configure the PPG3204 with the specified parameters for specified channels.

        Parameters
        ----------
        freq : :obj:`float`, optional
            Frequency of the pattern in Hz. The range is from 1.5 GHz to 32 GHz.
        patt_len : :obj:`int` or :obj:`Array_Like(int)`, optional
            Pattern length for every channel specified in ``CHs``.
        Vout : :obj:`float` or :obj:`Array_Like(float)`, optional
            Amplitud to set to the specify channels
        offset : :obj:`float` or :obj:`Array_Like(float)`, optional
            Offset to set to the specify channels
        bsh : :obj:`int` or :obj:`Array_Like(int)`, optional
            Bits shift to set to the specify channels
        skew : :obj:`float` or :obj:`Array_Like(float)`, optional
            Skew to set to the specify channels
        mode : :obj:`str`, optional
            Work mode of the PPG.
        order : :obj:`int` or :obj:`Array_Like(int)`, optional
            order of the polynomial generator. If ``mode='PRBS'``.
        data : :obj:`np.ndarray` or :obj:`Array_Like(np.ndarray)`, optional
            Data to set to the specify channels. If ``mode='DATA'``.
        CHs : :obj:`int` or :obj:`Array_Like(int)`, optional
            Channels to set the configuration.

        Examples
        --------
        In this examples we don't pass the argument ``addr_ID`` in order to print the commands output. For communication with a device this parameter is requered.

        .. code-block:: python

            >>> from opticomlib.lab import PPG3204
            >>> 
            >>> ppg = PPG3204()
            >>> ppg(freq=10e9, patt_len=1000, Vout=1.5, offset=0.5, bsh=10, skew=0.5e-12, mode='PRBS', order=7, CHs=2)
            :FREQ 1.0e+10
            :DIG2:PATT:LENG 1000
            :VOLT2:POS 1.5v
            :VOLT2:POS:OFFS 0.5v
            :DIG2:PATT:BSH 10
            :SKEW2 5e-13
            :DIG2:PATT:TYPE PRBS
            :DIG2:PATT:PLEN 7
        """
        if freq is not None:
            self.set_freq(freq)

        if patt_len is not None:
            self.set_patt_len(patt_len, CHs)
        
        if Vout is not None:
            self.set_output_voltage(Vout, CHs)

        if offset is not None:
            self.set_offset(offset, CHs)

        if bsh is not None:
            self.set_bits_shift(bsh, CHs)

        if skew is not None:
            self.set_skew(skew, CHs)

        if mode is not None:
            self.set_mode(mode, CHs)

        if order is not None and mode == 'PRBS':
            self.set_prbs_order(order, CHs)
        
        if data is not None and mode == 'DATA':
            self.set_data(data, CHs=CHs)
        return 'Done'
