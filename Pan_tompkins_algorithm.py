# Importing Libraries
import numpy as np
from scipy import signal as sg
from scipy.signal import butter, filtfilt


class Pan_tompkins:
    """ Implementationof Pan Tompkins Algorithm.
    Noise cancellation (bandpass filter) -> Derivative step -> Squaring and integration.
    Params:
        data (array) : ECG data
        sampling rate (int)
    returns:
        Integrated signal (array) : This signal can be used to detect peaks
    ----------------------------------------
    HOW TO USE ?
    Eg.
    ECG_data = [4, 7, 80, 78, 9], sampling = 2000
    
    call : 
       signal = Pan_tompkins(ECG_data, sampling).fit()
    ----------------------------------------
    
    """
    def __init__(self, data, sample_rate):

        self.data = data
        self.sample_rate = sample_rate


    def fit(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150, window_size=None):
        ''' Fit the signal according to algorithm and returns integrated signal
        
        '''
        # 1.Noise cancellationusing bandpass filter
        self.filtered_BandPass = self.band_pass_filter(normalized_cut_offs, butter_filter_order, padlen)
        
        # 2.derivate filter to get slpor of the QRS
        self.derviate_pass = self.derivative_filter()

        # 3.Squaring to enhance dominant peaks in QRS
        self.square_pass = self.squaring()

        # 4.To get info about QRS complex
        self.integrated_signal = self.moving_window_integration( window_size)

        return self.integrated_signal


    def band_pass_filter(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150):
        ''' Band pass filter for Pan tompkins algorithm
            with a bandpass setting of 5 to 20 Hz
            params:
                normalized_cut_offs (list) : bandpass setting canbe changed here
                bandpass filte rorder (int) : deffault 2
                padlen (int) : padding length for data , default = 150
                        scipy default value = 2 * max(len(a coeff, b coeff))
            return:
                filtered_BandPass (array)
        '''

        # Calculate nyquist sample rate and cutoffs
        nyquist_sample_rate = self.sample_rate / 2

        # calculate cutoffs
        if normalized_cut_offs is None:
            normalized_cut_offs = [5/nyquist_sample_rate, 15/nyquist_sample_rate]
        else:
            assert type(self.sample_rate ) is list, "Cutoffs should be a list with [low, high] values"

        # butter coefficinets 
        b_coeff, a_coeff = butter(butter_filter_order, normalized_cut_offs, btype='bandpass')[:2]

        # apply forward and backward filter
        filtered_BandPass = filtfilt(b_coeff, a_coeff, self.data, padlen=padlen)
        
        return filtered_BandPass


    def derivative_filter(self):
        ''' Derivative filter
        params:
            filtered_BandPass (array) : outputof bandpass filter
        return:
            derivative_pass (array)
        '''

        # apply differentiation
        derviate_pass= np.diff(self.band_pass_filter())

        return derviate_pass


    def squaring(self):
        ''' squaring application on derivate filter output data
        params:
        return:
            square_pass (array)
        '''

        # apply squaring
        square_pass= self.derivative_filter() **2

        return square_pass 


    def moving_window_integration(self, window_size=None):
        ''' Moving avergae filter 
        Params:
            window_size (int) : no. of samples to average, if not provided : 0.08 * sample rate
            sample_rate (int) : should be given if window_size is not given  
        return:
            integrated_signal (array)
        '''

        if window_size is None:
            assert self.sample_rate is not None, "if window size is None, sampling rate should be given"
            window_size = int(0.08 * int(self.sample_rate))  # given in paper 150ms as a window size
        

        # define integrated signal
        integrated_signal = np.zeros_like(self.squaring())

        # cumulative sum of signal
        cumulative_sum = self.squaring().cumsum()

        # estimationof area/ integral below the curve deifnes the data
        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)

        return integrated_signal


class HeartRateMaintainer():
	def __init__(self, signal, samp_freq, integrated_signal, band_pass_signal):
		'''
		Initialize Variables
		:param signal: input signal
		:param samp_freq: sample frequency of input signal
		'''

		# Initialize variables
		self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
		self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (0 for i in range(8))

		self.T_wave = False          
		self.m_win = integrated_signal
		self.b_pass = band_pass_signal
		self.samp_freq = samp_freq
		self.signal = signal
		self.win_150ms = round(0.15*self.samp_freq)

		self.RR_Low_Limit = 0
		self.RR_High_Limit = 0
		self.RR_Missed_Limit = 0
		self.RR_Average1 = 0


	def approx_peak(self):
		'''
		Approximate peak locations
		'''   

		# FFT convolution
		slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

		# Finding approximate peak locations
		for i in range(round(0.5*self.samp_freq) + 1,len(slopes)-1):
			if (slopes[i] > slopes[i-1]) and (slopes[i+1] < slopes[i]):
				self.peaks.append(i)  


	def adjust_rr_interval(self,ind):
		'''
		Adjust RR Interval and Limits
		:param ind: current index in peaks array
		'''

		# Finding the eight most recent RR intervals
		self.RR1 = np.diff(self.peaks[max(0,ind - 8) : ind + 1])/self.samp_freq   

		# Calculating RR Averages
		self.RR_Average1 = np.mean(self.RR1)
		RR_Average2 = self.RR_Average1
		
		# Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit  
		if (ind >= 8):
			for i in range(0, 8):
				if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit): 
					self.RR2.append(self.RR1[i])

					if (len(self.RR2) > 8):
						self.RR2.remove(self.RR2[0])
						RR_Average2 = np.mean(self.RR2)    

		# Adjusting the RR Low Limit and RR High Limit
		if (len(self.RR2) > 7 or ind < 8):
			self.RR_Low_Limit = 0.92 * RR_Average2        
			self.RR_High_Limit = 1.16 * RR_Average2
			self.RR_Missed_Limit = 1.66 * RR_Average2


	def searchback(self,peak_val,RRn,sb_win):
		'''
		Searchback
		:param peak_val: peak location in consideration
		:param RRn: the most recent RR interval
		:param sb_win: searchback window
		'''

		# Check if the most recent RR interval is greater than the RR Missed Limit
		if (RRn > self.RR_Missed_Limit):
			# Initialize a window to searchback  
			win_rr = self.m_win[peak_val - sb_win + 1 : peak_val + 1] 

			# Find the x locations inside the window having y values greater than Threshold I1             
			coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

			# Find the x location of the max peak value in the search window
			if (len(coord) > 0):
				for pos in coord:
					if (win_rr[pos] == max(win_rr[coord])):
						x_max = pos
						break
			else:
				x_max = None
 
			# If the max peak value is found
			if (x_max is not None):   
				# Update the thresholds corresponding to moving window integration
				self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI                         
				self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
				self.Threshold_I2 = 0.5 * self.Threshold_I1         

				# Initialize a window to searchback 
				win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) -1, x_max)]  

				# Find the x locations inside the window having y values greater than Threshold F1                   
				coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

				# Find the x location of the max peak value in the search window
				if (len(coord) > 0):
					for pos in coord:
						if (win_rr[pos] == max(win_rr[coord])):
							r_max = pos
							break
				else:
					r_max = None

				# If the max peak value is found
				if (r_max is not None):
					# Update the thresholds corresponding to bandpass filter
					if self.b_pass[r_max] > self.Threshold_F2:                                                        
						self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF                            
						self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
						self.Threshold_F2 = 0.5 * self.Threshold_F1      

						# Append the probable R peak location                      
						self.r_locs.append(r_max)                                                


	def find_t_wave(self,peak_val,RRn,ind,prev_ind):
		'''
		T Wave Identification
		:param peak_val: peak location in consideration
		:param RRn: the most recent RR interval
		:param ind: current index in peaks array
		:param prev_ind: previous index in peaks array
		'''

		if (self.m_win[peak_val] >= self.Threshold_I1): 
			if (ind > 0 and 0.20 < RRn < 0.36):
				# Find the slope of current and last waveform detected        
				curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms/2) : peak_val + 1]))
				last_slope = max(np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms/2) : self.peaks[prev_ind] + 1]))
				
				# If current waveform slope is less than half of last waveform slope
				if (curr_slope < 0.5*last_slope):  
					# T Wave is found and update noise threshold                      
					self.T_wave = True                             
					self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI 

			if (not self.T_wave):
				# T Wave is not found and update signal thresholds
				if (self.probable_peaks[ind] > self.Threshold_F1):   
					self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI                                         
					self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF 

					# Append the probable R peak location
					self.r_locs.append(self.probable_peaks[ind])  

				else:
					self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI
					self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF                   

		# Update noise thresholds
		elif (self.m_win[peak_val] < self.Threshold_I1) or (self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
			self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI  
			self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF


	def adjust_thresholds(self,peak_val,ind):
		'''
		Adjust Noise and Signal Thresholds During Learning Phase
		:param peak_val: peak location in consideration
		:param ind: current index in peaks array
		'''

		if (self.m_win[peak_val] >= self.Threshold_I1): 
			# Update signal threshold
			self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI

			if (self.probable_peaks[ind] > self.Threshold_F1):                                            
				self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF 

				# Append the probable R peak location
				self.r_locs.append(self.probable_peaks[ind])  

			else:
				# Update noise threshold
				self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF                                    
        
		# Update noise thresholds    
		elif (self.m_win[peak_val] < self.Threshold_I2) or (self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
			self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI  
			self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF


	def update_thresholds(self):
		'''
		Update Noise and Signal Thresholds for next iteration
		'''

		self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
		self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
		self.Threshold_I2 = 0.5 * self.Threshold_I1 
		self.Threshold_F2 = 0.5 * self.Threshold_F1
		self.T_wave = False 


	def ecg_searchback(self):
		'''
		Searchback in ECG signal to increase efficiency
		'''

		# Filter the unique R peak locations
		self.r_locs = np.unique(np.array(self.r_locs).astype(int))

		# Initialize a window to searchback
		win_200ms = round(0.2*self.samp_freq)
	
		for r_val in self.r_locs:
			coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

			# Find the x location of the max peak value
			if (len(coord) > 0):
				for pos in coord:
					if (self.signal[pos] == max(self.signal[coord])):
						x_max = pos
						break
			else:
				x_max = None

			# Append the peak location
			if (x_max is not None):   
				self.result.append(x_max)


	def find_r_peaks(self):
		'''
		R Peak Detection
		'''

		# Find approximate peak locations
		self.approx_peak()

		# Iterate over possible peak locations
		for ind in range(len(self.peaks)):

			# Initialize the search window for peak detection
			peak_val = self.peaks[ind]
			win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms), min(self.peaks[ind] + self.win_150ms, len(self.b_pass)-1), 1)
			max_val = max(self.b_pass[win_300ms], default = 0)

			# Find the x location of the max peak value
			if (max_val != 0):        
				x_coord = np.asarray(self.b_pass == max_val).nonzero()
				self.probable_peaks.append(x_coord[0][0])
			
			if (ind < len(self.probable_peaks) and ind != 0):
				# Adjust RR interval and limits
				self.adjust_rr_interval(ind)
				
				# Adjust thresholds in case of irregular beats
				if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit): 
					self.Threshold_I1 /= 2
					self.Threshold_F1 /= 2

				RRn = self.RR1[-1]

				# Searchback
				self.searchback(peak_val,RRn,round(RRn*self.samp_freq))

				# T Wave Identification
				self.find_t_wave(peak_val,RRn,ind,ind-1)

			else:
				# Adjust threholds
				self.adjust_thresholds(peak_val,ind)

			# Update threholds for next iteration
			self.update_thresholds()

		# Searchback in ECG signal 
		self.ecg_searchback()

		return self.result