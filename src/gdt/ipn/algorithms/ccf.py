#  CONTAINS TECHNICAL DATA/COMPUTER SOFTWARE DELIVERED TO THE U.S. GOVERNMENT WITH UNLIMITED RIGHTS
#
#  Contract No.: CA 80NSSC24M0035
#  Contractor Name: Universities Space Research Association
#  Contractor Address: 7178 Columbia Gateway Drive, Columbia, MD 21046
#
#  Copyright 2021-2025 by Universities Space Research Association (USRA). All rights reserved.
#
#  Original IPN development funded through FY21 USRA Internal Research and Development Funds
#  and FY21 NASA-MSFC Center Innovation Funds
#
#  IPN code developed by:
#
#                Corinne Fletcher, Rachel Hamburg and Adam Goldstein
#                Universities Space Research Association
#                Science and Technology Institute
#                https://sti.usra.edu
#
#                Peter Veres
#                University of Alabama in Huntsville
#                Huntsville, AL
#
#                Michelle Hui
#                National Aeronautics and Space Administration (NASA)
#                Marshall Space Flight Center
#                Astrophysics Branch (ST-12)
#
#
#  With code contributions by:
#
#                Dmitry Svinkin
#                Ioffe Institute
#                St. Petersburg, Russia
#
#  Included in the Gamma-Ray Data Toolkit
#  Copyright 2017-2025 by Universities Space Research Association (USRA). All rights reserved.
#
#  Developed by: William Cleveland and Adam Goldstein
#                Universities Space Research Association
#                Science and Technology Institute
#                https://sti.usra.edu
#
#  Developed by: Daniel Kocevski
#                National Aeronautics and Space Administration (NASA)
#                Marshall Space Flight Center
#                Astrophysics Branch (ST-12)
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
#  in compliance with the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#  implied. See the License for the specific language governing permissions and limitations under the
#  License.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from gdt.core.healpix import HealPixLocalization as hpl 

from ..instrument import Spacecraft, TimeUncertainty
from .annulus import Annulus, IpnHealPixLocalization

class Localization:
    def __init__(self):
        self._spacecraft = []
    
    def _validate_spacecraft(self, spacecraft):
        if not isinstance(spacecraft, Spacecraft):
            raise TypeError("Input must be of 'Spacecraft' class")
    
    @classmethod
    def from_list(cls, spacecraft_list):
        obj = cls()
        [obj._validate_spacecraft(sc) for sc in spacecraft_list]
        obj._spacecraft = spacecraft_list
        return obj
        
    def add_spacecraft(self, spacecraft):
        self._validate_spacecraft(spacecraft)
        self._spacecraft.append(spacecraft)
    
    def localize(self):
        pass

    def get_healpix(self):
        pass        

class Ipn(Localization):
    """The localization algorithm used by the current IPN. 
    
    This algorithm can only work with two spacecraft at one time.  Each
    lightcurve is background subtracted, and then one of the lightcurves is
    normalized to the number of counts of the other lightcurve. A source
    window in one lightcurve is assumed and slices of the same source window
    duration from the second lightcurve are shifted in time relative to each 
    other. At each shift the cross-correlation and chi-squared-like statistic
    between the two lightcurves are calculated. The minimum (maximum) of the 
    chi-squared-like statistic (cross-correlation) is taken as the time offset
    between the two lightcurves for the localization.

    Reference:
    `Pal'shin, V. D., Hurley, K., Svinkin, D. S., et al., 2013, ApJS, 207, 2
    <https://iopscience.iop.org/article/10.1088/0067-0049/207/2/38>`_ 
    """
    def __init__(self):
        super().__init__()
        self._time_offset = None

    @property
    def time_offset(self):
        return self._time_offset
    
    @classmethod
    def from_list(cls, spacecraft_list):
        """Create a new object from a list of spacecraft. No more than two 
        spacecraft can be used.
        
        Args:
            spacecraft_list (list): The list of :class:`Spacecraft`
        
        Returns:
            (:class:`ClassicIpn`)
        """
        if len(spacecraft_list) > 2:
            raise ValueError("CurrentIpn can only work with 2 spacecraft")
        return super().from_list(spacecraft_list)
    
    def max_time_offset(self):
        """The maximum light travel time between the two spacecraft.
        
        Returns:
            (float): maximum light travel time
        """
        return self._spacecraft[0].position.light_travel_time( \
                                                self._spacecraft[1].position)

    def _set_lightcurves(self):
        """Set up the lightcurves for cross-correlation."""
        lc1_obs, lc2_obs = [sc.observation for sc in self._spacecraft]
        dt1 = lc1_obs.data.lo_edges[1] - lc1_obs.data.lo_edges[0]
        dt2 = lc2_obs.data.lo_edges[1] - lc2_obs.data.lo_edges[0]

        lc1, lc2 = self._get_background_subtracted_lightcurves(lc1_obs, lc2_obs)
        self._set_lightcurve_attributes(lc1, lc2, lc1_obs.data, lc2_obs.data, dt1, dt2)
        return
    
    def _validate_lightcurve_length(self, src_interval, max_offset):
        """Checks the duration of the lightcurves and ensures there is enough
        data to perform the cross-correlation

        Args:
            src_interval (tuple): the source interval time selection
            max_offset (float): the maximum time offset considered
        """
        lower_bound = src_interval[0] - max_offset
        upper_bound = src_interval[1] + max_offset
        if lower_bound < self._times2[0] or upper_bound > self._times2[-1]:
            raise ValueError("Need more data in lightcurve 2 to perform cross-correlation")
        return
    
    def _get_background_subtracted_lightcurves(self, lc1_full, lc2_full):
        """Try to subtract background; fall back to full lightcurves if needed
        
        Args:
            lc1_full (gdt.ipn.instrument.Observation): the first lightcurve
            lc2_full (gdt.ipn.instrument.Observation): the second lightcurve

        Returns:
            (gdt.core.data_primitives.TimeBins, gdt.core.data_primitives.TimeBins):
                the background-subtracted lightcurves
        """
        try:
            lc1 = lc1_full.background_subtract()
            lc2 = lc2_full.background_subtract()
        except Exception as e:
            warnings.warn("Cannot subtract background from lightcurves." \
                        "Using full lightcurves instead.")
            lc1 = lc1_full.data
            lc2 = lc2_full.data
        return lc1, lc2

    def _set_lightcurve_attributes(self, lc1, lc2, lc1_full, lc2_full, dt1, dt2):
        """Set internal attributes for time, counts, errors, and time resolution.
        If lightcurve 2 has a larger binning that lightcurve 1, then the lightcurves
        are switched, so that the reference lightcurve has the larger binning
        
        Args:
            lc1 (gdt.core.data_primitives.TimeBins): the first lightcurve, background-subtracted
            lc2 (gdt.core.data_primitives.TimeBins): the second lightcurve, background-subtracted
            lc1_full (gdt.core.data_primitives.TimeBins): the first lightcurve
            lc2_full (gdt.core.data_primitives.TimeBins): the second lightcurve
        """
        if dt2 > dt1:
            self._times1 = lc2.lo_edges
            self._times2 = lc1.lo_edges
            self._counts1 = lc2.counts
            self._counts2 = lc1.counts
            self._err1 = lc2_full.counts
            self._err2 = lc1_full.counts
            self._dt1 = dt2
            self._dt2 = dt1
            self._switch = True 
        else:
            self._times1 = lc1.lo_edges
            self._times2 = lc2.lo_edges
            self._counts1 = lc1.counts
            self._counts2 = lc2.counts
            self._err1 = lc1_full.counts
            self._err2 = lc2_full.counts
            self._dt1 = dt1
            self._dt2 = dt2
            self._switch = False
        return

    def _shift_array(self, max_dt):
        """Create lightcurve shift array
        """
        num_shifts = int(np.abs(max_dt) / min(self._dt1, self._dt2))
        shift_array = np.arange(num_shifts, dtype=int)
        return np.concatenate((shift_array - num_shifts, [0], shift_array + 1))

    def _scale_factor(self, src, times1, times2, counts1, counts2):
        """The normalization between lightcurves from instruments 
        with different count rates.

        Returns:
            (float): scale factor applied to 
        """
        slice1 = np.logical_and(times1 >= src[0], times1 <= src[1])
        slice2 = np.logical_and(times2 >= src[0], times2 <= src[1])
        return (counts1[slice1].sum()) / counts2[slice2].sum()

    def localize(self, src, max_dt=0., plot=False):
        """Cross-correlate two light curves using scipy's correlate function.
        The second lightcurve is shifted wrt the first lightcurve.
        
        Args:
            src (tuple): the start and stop times of source interval in lightcurve 1
            lc2_start (float): the start time of the source interval in lightcurve 2
            max_dt (float): the maximum time lag (1-sided) to consider, 
                            e.g., the light-travel-time between spacecraft
            plot (Boolean): display lightcurves for each time shift
        """
        # Set lightcurves and check that there is enough data
        self._set_lightcurves()
 
        # set the max time offset
        if max_dt == 0.:
            max_dt = self.max_time_offset()[0]
        
        # check lightcurve 2 length
        self._validate_lightcurve_length(src, max_dt)

        # calculate scale over selection of background-subtracted lightcurve 
        # and apply to source+background lightcurves
        scale = self._scale_factor(src, self._times1, self._times2, 
                                        self._counts1, self._counts2)
        counts2 = self._counts2 * scale
        err2 = self._err2 * scale ** 2
 
        # slice lightcurve 1 so that it only includes source region
        mask = (self._times1 >= src[0]) & (self._times1 <= src[1])
        self._times1_cut = self._times1[mask]
        self._counts1_cut = self._counts1[mask]
        self._err1_cut = self._err1[mask]

        # shift lightcurves
        shift_array = self._shift_array(max_dt)
        self._chi2, self._ccf = self.shift(
            shift_array, src, counts2, err2, plot=plot)
        if self._switch is not False:
            self._chi2 = self._chi2[::-1]
            self._ccf = self._ccf[::-1]

        # Get the timing uncertainties
        self._dts = shift_array * self._dt2
        self._dt_min = self._dts[np.argmin(self._chi2)]
        self._dt_lo, self._dt_hi = self.chi2_confidence_interval(
            sigma=3., dof=len(self._counts1_cut[:-1]))
        self._time_offset = TimeUncertainty(
            self._dt_min, (self._dt_min-self._dt_lo, self._dt_hi-self._dt_min))
        return

    def shift(self, shift_array, src, counts2, err2, plot=False):
        """Shift lightcurve 2 with respect to lightcurve 1

        Args:
            shift_array (np.array): the list of time shifts 
            src (tuple): the amount of data to slice in each lightcurve
            plot (Boolean): plot the lightcurves at each shift;
                default is False

        Returns:
            (np.array, np.array): the chi-squared statistic per degrees 
                of freedom and the cross-correlation statistic for each
                time shift
        """
        chisq = []
        ccf = []
        for i, shift in enumerate(shift_array):
            start = src[0] + (shift * self._dt2)
            end = start + (src[1] - src[0])
       
            # slice lightcurve 2
            #mask = (self._times2 >= start) & (self._times2 < end + self._dt2)
            if shift < 0:
                mask = (self._times2 >= start) & (self._times2 < end + self._dt2)
            elif shift > 0:
                mask = (self._times2 > start - self._dt2) & (self._times2 <= end)
            else:
                mask = (self._times2 >= start) & (self._times2 <= end)
            times2 = self._times2[mask]
            counts2_tmp = counts2[mask]
            err2_tmp = err2[mask]
            
            # define the new bin edges for lightcurve 2
            cum_diff = np.cumsum(np.diff(self._times1_cut))
            hi_bin_edges = [times2[0] + d for d in np.cumsum(np.diff(self._times1_cut))]
            new_bin_edges = np.insert(hi_bin_edges, 0, times2[0])
            new_bin_edges = [np.round(e, 10) for e in new_bin_edges]
       
            # bin lightcurve 2 to match lightcurve 1
            rebinned_counts2 = []
            rebinned_err2 = []
            for b in range(len(new_bin_edges)-1):
                bins = (times2 >= new_bin_edges[b]) & (times2 < new_bin_edges[b+1])
                times = times2[bins]
                rebinned_counts2.append(np.sum(counts2_tmp[bins]))
                rebinned_err2.append(np.sum(err2_tmp[bins]))
            rebinned_counts2 = np.array(rebinned_counts2)
            rebinned_err2 = np.array(rebinned_err2)

            if plot is not False:
                self.plot_shift(shift, rebinned_counts2)
        
            # calculate the chi2 and ccf
            x2 = self.chisq(self._counts1_cut[:-1], rebinned_counts2, 
                    variance1=self._err1_cut[:-1], variance2=rebinned_err2,
                    dof=len(rebinned_counts2)-1)
            chisq.append(x2)
            ccf.append(self.ccf(self._counts1_cut[:-1], rebinned_counts2))
        return chisq, ccf

    def chisq(self, counts1, counts2, variance1=None, variance2=None, dof=1):
        """Chi-squared statistic for the comparison of `counts1` to `counts2`
        
        Args:
            counts1 (np.array): The first lightcurve background-subtracted counts
            counts2 (np.array): The second lightcurve background-subtracted counts
            variance1 (np.array): The variance of the first lightcurve
            variance2 (np.array): The variance of the second lightcurve
            dof (int): The degrees of freedom

        Returns:
            (float): The chi-squared statistic per degrees of freedom
        """
        mask = (counts1 > 0) & (counts2 > 0)
        if variance1 is not None and variance2 is not None:
            r = (counts1[mask] - counts2[mask])**2/ \
                (variance1[mask] + variance2[mask])
        else:
            r = (counts1[mask] - counts2[mask])**2/ \
                (counts1[mask] + counts2[mask])
        return r.sum() / dof

    def ccf(self, counts1, counts2):
        """The cross-correlation of `counts1` and `counts2`
        
        Args:
            counts1 (np.array): The first lightcurve counts
            counts2 (np.array): The second lightcurve counts

        Returns:
            (float): The cross-correlation
        """
        norm1 = counts1 - np.mean(counts1)
        norm2 = counts2 - np.mean(counts2)
        return np.sum(norm1 * norm2) / np.sqrt(np.sum(norm1**2) * np.sum(norm2**2))

    def chi2_confidence_interval(self, sigma=3., dof=1):
        """
        Compute the ±nSigma confidence interval on dT using the chi-squared 
        distribution. Based on Eq (5) of Pal'shin et al. (2013).

        Args:
            sigma (float): the confidence level
            dof (int): the degrees of freedom

        Returns:
            (float, float): the lower and upper time lag uncertainties
        """
        # Calculate threshold chi-squared value (reduced chi^2)
        p_0 = stats.norm.cdf(sigma) - stats.norm.cdf(-sigma)
        self._chi2_lim = stats.chi2.ppf(p_0, dof - 1) / (dof - 1) - 1 + min(self._chi2)
        chisq_lim_idx = np.where(self._chi2 <= self._chi2_lim)[0]
    
        # Find the indices of the upper and lower dT limits 
        if len(chisq_lim_idx) == len(self._chi2):
            idx_lo = chisq_lim_idx[0]
            idx_hi = chisq_lim_idx[-1]
        else:
            if chisq_lim_idx[0] == 0:
                idx_lo = chisq_lim_idx[0]
            else:
                idx_lo = chisq_lim_idx[0] - 1
            if chisq_lim_idx[-1] == len(self._chi2)-1:
                idx_hi = chisq_lim_idx[-1]
            else:
                idx_hi = chisq_lim_idx[-1] + 1

        # Get time lag values 
        return self._dts[idx_lo], self._dts[idx_hi]

    def get_healpix(self, nside=2048):
        """Return the localization as HealPix object
        
        Returns:
            (:class:`HealPixLocalization`)
        """
        if self.time_offset is None:
            raise RuntimeError('Localization has not yet been performed')

        annulus = Annulus(*self._spacecraft, self._time_offset)
        return IpnHealPixLocalization.from_annulus(
            *annulus.center(), annulus.radius(), 
            annulus.total_width(), nside=nside
        )

    def plot_shift(self, i_shift, new_counts):
        """
        Plots the shifted lightcurves

        Args:
            i_shift (int): the time lag shift number
            new_counts (array): the shifted (and potentially rebinned)
                counts from the second lightcurve

        Returns: None
        """
        plt.figure(figsize=(8, 5))

        # stationary lightcurve 1
        times1 = np.append(self._times1, self._times1[-1] + self._dt1)
        plt.stairs(self._counts1, times1, color='lightsteelblue') 
        plt.stairs(self._counts1_cut[:-1], self._times1_cut, 
            color='C0', linewidth=2, label='Lightcurve 1')

        # shifted lightcurve 2
        plt.stairs(new_counts, self._times1_cut, 
            color='C1', linewidth=2, label='Lightcurve 2 (rebinned)')

        # styling
        plt.xlim(left=self._times1_cut[0] - 10 * self._dt1,
                 right=self._times1_cut[-1] + 10 * self._dt1)
        shift_time = np.round(i_shift * self._dt2, 5)
        plt.title('Shift {} ({:.4f}) s'.format(i_shift, shift_time))
        plt.xlabel('Time (s)')
        plt.ylabel('Background subtracted counts')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
        return

    def plot_fit(self):
        """Produce a plot of the chi-squared and 
        cross-correlation statistics
        """
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.axvspan(self._dt_lo, self._dt_hi, color='grey', alpha=0.50)
        ax1.plot(self._dts, self._ccf, color='C0', marker='o')
        ax1.plot(self._dts, self._ccf, color='C0', linestyle='--')
        ax1.set_xlabel('Time lag [s]', fontsize=12)
        ax1.set_ylabel('Cross-Correlation Function', fontsize=12, color='C0')
        
        ax2 = ax1.twinx()
        ax2.plot(self._dts, self._chi2, color='C1', marker='o')
        ax2.plot(self._dts, self._chi2, color='C1')
        ax2.axvline(x=self._dt_min, color='grey', linestyle='--')
        ax2.set_ylabel('Reduced Chi-Squared', fontsize=12, color='C1')
        fig.tight_layout()
        plt.show()
        plt.close()
        return
        

class ClassicIpn(Localization):
    """The "Classic" IPN localization algorithm used by the first IPN.
     
    This algorithm can only work with two spacecraft at one time.  Each
    lightcurve is background subtracted, and then one of the lightcurves is
    normalized to the number of counts of the other lightcurve.  One of the 
    lightcurves is then shifted in time relative to the other, bin by bin, and 
    at each shift the cross-correlation and chi-squared between the two 
    lightcurves are calculated.  Finally, a quadratic function is fit to both
    the cross-correlation function and chi-squared as a function of time, and
    the maximum/minimum of the fitted function is used as the time offset 
    between the two lightcurves for the localization.
    
    References:
        `Hurley, K., Briggs, M. S., Kippen, R. M., et al., 1999, ApJS, 120, 399 
        <https://iopscience.iop.org/article/10.1086/313178>`_
    """
    def __init__(self):
        super().__init__()
        self.fitter = None
        self.time_offset = None
    
    @classmethod
    def from_list(cls, spacecraft_list):
        """Create a new object from a list of spacecraft. No more than two 
        spacecraft can be used.
        
        Args:
            spacecraft_list (list): The list of :class:`Spacecraft`
        
        Returns:
            (:class:`ClassicIpn`)
        """
        if len(spacecraft_list) > 2:
            raise ValueError("ClassicIpn can only work with 2 spacecraft")
        return super().from_list(spacecraft_list)
    
    def add_spacecraft(self, spacecraft):
        """Add a new spacecraft.  Note that the object cannot have more than
        two spacecraft.
        
        Args:
            spacecraft (:class:`Spacecraft`): The spacecraft to add to the 
                                              algorithm
        """
        if len(self._spacecraft) == 2:
            raise ValueError("ClassicIpn can only work with 2 spacecraft")
        super().add_spacecraft(spacecraft)
    
    def max_dt(self):
        """The maximum number of lightcurve bin shifts as determined by the 
        light travel time between the two spacecraft.
        
        Returns:
            (int): The number of bin shifts
        """
        return self._spacecraft[0].position.light_travel_time( \
                                                self._spacecraft[1].position)
        
    def normalize_counts(self, counts1, counts2):
        """Normalizes the array of counts in `counts2` to the total number of 
        counts in `counts1`.
        
        Args:
            counts1 (np.array): The first lightcurve counts
            counts2 (np.array): The second lightcurve counts
        
        Returns:
            (np.array): `counts2` normalized to `counts1`
        """
        ratio = counts1.sum()/counts2.sum()
        return counts2*ratio
        
    def chisq(self, counts1, counts2):
        """Chi-squared statistic for the comparison of `counts1` to `counts2`
        
        Args:
            counts1 (np.array): The first lightcurve counts
            counts2 (np.array): The second lightcurve counts

        Returns:
            (float): The chi-squared statistic
        """
        mask = (counts1 > 0) & (counts2 > 0)
        r = (counts1[mask] - counts2[mask])**2/(counts1[mask] + counts2[mask])
        return r.sum()
    
    def ccf(self, counts1, counts2):
        """The cross-correlation of `counts1` and `counts2`.
        
        Args:
            counts1 (np.array): The first lightcurve counts
            counts2 (np.array): The second lightcurve counts

        Returns:
            (float): The cross-correlation
        """
        norm1 = counts1 - counts1.mean()
        norm2 = counts2 - counts2.mean()
        corr = np.sum(norm1 * norm2) / np.sqrt(np.sum(norm1**2) * \
                                               np.sum(norm2**2))
        return corr
      
    def localize(self):
        """Perform the localization on the observations of the two spacecraft
        """
        max_dt = self.max_dt()[0]
        
        # background subtract lightcurves
        source_lcs = [sc.observation.background_subtract() \
                      for sc in self._spacecraft]
        
        # normalize counts in each lightcurve
        counts1 = source_lcs[0].counts
        counts2 = self.normalize_counts(counts1, source_lcs[1].counts)
        times1 = source_lcs[0].lo_edges
        times2 = source_lcs[1].lo_edges

        # calculate the CCF and Chisq for each bin shift
        bin_shifter = BinShifter(counts1, counts2, times1, times2, max_dt=max_dt)
        ccf = []
        chisq = []
        for shifted_arrays in bin_shifter:
            ccf.append(self.ccf(*shifted_arrays))
            chisq.append(self.calc_chisq(*shifted_arrays))
        ccf = np.array(ccf)
        chisq = np.array(chisq)

        # quadratic fit to CCF to determine time lag and uncertainty
        bin_widths = self._spacecraft[0].observation.data.widths[0]
        self.fitter = CcfQuadraticFit(bin_shifter.shift_array*bin_widths, ccf,
                                      chisq)
        self.fitter.do_fit()
        self.time_offset = TimeUncertainty(self.fitter.time_lag(), 
                                           (self.fitter.time_lag_uncert(0.683),
                                           self.fitter.time_lag_uncert(0.683)))

    def get_healpix(self, nside=2048):
        """Return the localization as HealPix object
        
        Returns:
            (`:class:`HealPixLocalization`)
        """
        if self.time_offset is None:
            raise RuntimeError('Localization has not yet been performed')
        annulus = Annulus(*self._spacecraft, self.time_offset)
        return hpl.from_annulus(*annulus.center(), annulus.radius(), 
                                annulus.total_width()[0], nside=nside) 

    def plot_fit(self, clevel=0.683):
        """Produce a plot of the quadratic fit to the CCF and chi-squared
        statistic.  The maximum likelihood time offset is shown by the 
        gray vertical bar
        
        Args:
            clevel (float, optional): The confidence level. Default is 0.683
        """
        time_arr = np.linspace(self.fitter.times[0], self.fitter.times[-1], 100)
        ccf_fit = self.fitter.eval_quadratic(time_arr, self.fitter.coeffs_ccf)
        chisq_fit = self.fitter.eval_quadratic(time_arr, self.fitter.coeffs_chisq) #/chisq_factor

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(self.fitter.times, self.fitter.ccf, color='C0', marker='o', label='CCF')
        ax1.plot(time_arr, ccf_fit, color='C0', linestyle='--')
        ax1.set_xlabel('Time [s]', fontsize=12)
        ax1.set_ylabel('Cross-Correlation Function', fontsize=12, color='C0')
        ax1.set_ylim(0.0, 1.0)

        ax2 = ax1.twinx()
        ax2.plot(self.fitter.times, self.fitter.chisq, color='C1', marker='o', label='Chisq')
        ax2.plot(time_arr, chisq_fit, color='C1', linestyle='--')
        ax2.set_ylabel('Chi-Squared', fontsize=12, color='C1')
        ax2.set_ylim(0.9*self.fitter.chisq.min(), 1.1*self.fitter.chisq.max())

        lag = self.fitter.time_lag()
        uncert = self.fitter.time_lag_uncert(clevel)
        ax1.axvspan(lag-uncert, lag+uncert, color='black', alpha=0.5)

        fig.tight_layout()
        plt.show()
        plt.close()

class CcfQuadraticFit:
    """Perform a quadratic fit to cross-correlation and chi-squared curves.
    The curves may not be purely quadratic, so a threshold cut on the 
    chi-squared curve is applied.  The remaining points in the curves are 
    recursively fit, each time removing the point that most contributes to the 
    chi-squared of the fit, until the removal of a point does not significantly
    improve the fit.  The time lag and uncertainty can then be estimated from
    the quadratic fit.
    
    Parameters:
        times (np.array): The array of times
        ccf (np.array): The values of the cross-correlation function
        chisq (np.array): The values of the chi-squared statistic
    """
    def __init__(self, times, ccf, chisq):
        self.times = times
        self.ccf = ccf
        self.chisq = chisq
        self.coeffs_chisq = np.array([]*times.size)
        self.coeffs_ccf = np.array([]*times.size)
        
    def good_points_mask(self, rel_max_chisq=0.75):
        """A mask into the data arrays that only return points that are 
        less than, or equal to, the relative fraction of the maximum chi-squared
        
        Args:
            rel_max_chisq (float, optional): Relative fraction of the maximum
                                             chi-squared.  Default is 0.75.
        """
        max_chisq = self.chisq.max()
        min_chisq = self.chisq.min()
        thresh = (max_chisq-min_chisq) * rel_max_chisq
        mask = (self.chisq < thresh)
        return mask
      
    def do_fit(self, **kwargs):
        """Perform the fit by recursively removing the point that most 
        contributes to the fit statistic until the fit stops significantly 
        improving.
        
        Args:
            rel_max_chisq (float, optional): Relative fraction of the maximum
                                             chi-squared.  Default is 0.75.
        """
        mask = self.good_points_mask(**kwargs)
        old_times = self.times[mask]
        old_ccf = self.ccf[mask]
        old_chisq = self.chisq[mask]
        # perform the fits
        coeffs_chisq, cov_chisq = np.polyfit(old_times, old_chisq, 2, cov=True)
        coeffs_ccf, cov_ccf = np.polyfit(old_times, old_ccf, 2, cov=True)
        # the chisq of each point and reduced chisq
        chisq_pt = self.calc_chisq(old_times, old_chisq, coeffs_chisq) + \
                   self.calc_chisq(old_times, old_ccf, coeffs_ccf)
        redchisq = chisq_pt.sum()/(2*old_times.size)
        
        while (True):
            new_mask = (chisq_pt != chisq_pt.max())
            new_times = old_times[new_mask]
            new_ccf = old_ccf[new_mask]
            new_chisq = old_chisq[new_mask]
            # perform the fits
            new_coeffs_chisq, new_cov_chisq = np.polyfit(new_times, new_chisq, 
                                                         2, cov=True)
            new_coeffs_ccf, new_cov_ccf = np.polyfit(new_times, new_ccf, 2, 
                                                     cov=True)
            # the chisq of each point and reduced chisq
            new_chisq_pt = self.calc_chisq(new_times, new_chisq, 
                                           new_coeffs_chisq) + \
                           self.calc_chisq(new_times, new_ccf, new_coeffs_ccf)
            new_redchisq = new_chisq_pt.sum()/(2*new_times.size)
            
            # if new reduced chisq is not significantly better than previous,
            # then stop, otherwise continue to next iteration
            if np.abs(new_redchisq-1.0) < np.abs(redchisq-1.0):    
                old_times = new_times
                old_ccf = new_ccf
                old_chisq = new_chisq
                coeffs_chisq = new_coeffs_chisq
                cov_chisq = new_cov_chisq
                coeffs_ccf = new_coeffs_ccf
                cov_ccf = new_cov_ccf
                chisq_pt = new_chisq_pt
                redchisq = new_redchisq
            else:
                break
            
            if old_times.size == 4:
                break

        self.coeffs_chisq = coeffs_chisq
        self.coeffs_ccf = coeffs_ccf
    
    def eval_quadratic(self, times, coeffs):
        """Evaluate the quadratic function
        
        Args:
            times (np.array): The times array
            coeffs (np.array): The quadratic coefficients

        Returns:
            (np.array)
        """
        return coeffs[0]*times**2 + coeffs[1]*times + coeffs[2]
    
    def calc_chisq(self, times, ccf, coeffs):
        """Calculate the chi-squared contribution of each point
        
        Args:
            times (np.array): The times array
            ccf (np.array): The cross-correlation function values
            coeffs (np.array): The quadratic coefficients
        
        Returns:
            (np.array)
        """
        fxn = self.eval_quadratic(times, coeffs)
        return ((fxn-ccf)**2)
    
    def time_lag(self):
        """Calculate the time lag, defined as the location of the 
        maximum/minimum of the quadratic
        
        Returns:
            (float)
        """
        coeffs = self.coeffs_chisq
        return -coeffs[1]/(2.0*coeffs[0])
        
    def time_lag_uncert(self, clevel):
        """Calculate the time lag uncertainty for the given confidence level.
        Determined from the fit to the chi-squared curve.
        
        Args:
            clevel (float): The confidence level
        
        Returns:
            (float)
        """
        coeffs = self.coeffs_chisq
        chi2_diff = stats.chi2.ppf(clevel, 1)
        time_lag = self.time_lag()
        min_chi2 = self.eval_quadratic(time_lag, coeffs)
        chi2 = min_chi2+chi2_diff
        uncert = (-coeffs[1] + np.sqrt(4.0*coeffs[0] * (chi2-coeffs[2]) + \
                                       coeffs[1]**2)) / (2.0*coeffs[0])
        return np.abs(time_lag-uncert)  

    def plot_fit(self, clevel=0.683):
        """Produce a plot of the quadratic fit to the CCF and chi-squared
        statistic.  The maximum likelihood time offset is shown by the 
        gray vertical bar
        
        Args:
            clevel (float, optional): The confidence level. Default is 0.683
        """
        chisq_factor = self.chisq.max()
        time_arr = np.linspace(self.times[0], self.times[-1], 100)
        ccf_fit = self.eval_quadratic(time_arr, self.coeffs_ccf)
        chisq_fit = self.eval_quadratic(time_arr, self.coeffs_chisq)#/chisq_factor
        
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(self.times, self.ccf, color='C0', marker='o', label='CCF')
        ax1.plot(time_arr, ccf_fit, color='C0', linestyle='--')
        ax1.set_xlabel('Time [s]', fontsize=12)        
        ax1.set_ylabel('Cross-Correlation Function', fontsize=12, color='C0')
        ax1.set_ylim(0.0, 1.0)
        
        ax2 = ax1.twinx()
        ax2.plot(self.times, self.chisq, color='C1', marker='o', label='Chisq')
        ax2.plot(time_arr, chisq_fit, color='C1', linestyle='--')
        ax2.set_ylabel('Chi-Squared', fontsize=12, color='C1')
        ax2.set_ylim(0.9*self.chisq.min(), 1.1*self.chisq.max())
        
        lag = self.time_lag()
        uncert = self.time_lag_uncert(clevel)
        ax1.axvspan(lag-uncert, lag+uncert, color='black', alpha=0.5)
        
        fig.tight_layout()
        plt.show()   
        plt.close()
   
class BinShifter:
    """Consecutively shifts two binned arrays against each other.

    Shifts array2 relative to array1 by a range of bin shifts, trimming
    both arrays to preserve only overlapping regions.

    Parameters:
        counts1, counts2 (np.array): Count arrays
        times1, times2 (np.array): Time arrays (aligned with counts)
        max_dt (float): Maximum time shift allowed in seconds
        plot (bool): Unused; placeholder for future visualization
    """
    def __init__(self, counts1, counts2, times1, times2, max_dt=0., plot=False):
        self._max_dt = max_dt
        self._plot = plot

        self._set_arrays(times1, counts1, times2, counts2)
        self._check_arrays()
        self._shift_array = self._calc_shift_array()
        self._shift_slices = self._calc_shift_slices()
        self._iter = 0

    def __iter__(self):
        self._iter = 0
        return self
    
    def __next__(self):
        if self._iter < self.shift_array.size:
            arr1, arr2 = self.shift(self._iter)
            self._iter += 1
            return (arr1[1], arr2[1]) # return counts only
        else:
            raise StopIteration
    
    def _calc_shift_array(self):
        # the shift array is num_shifts in negative direction, num_shifts in
        # the positive direction, and the 0 shift in the middle
        if self._num_shifts == 0:
            return np.array([0], dtype=int)
        arr = np.arange(self._num_shifts)
        arr = np.concatenate((arr-self._num_shifts, [0.0], arr+1))
        return arr.astype(int)

    def _set_arrays(self, times1, counts1, times2, counts2):
        """Setting Array1 and Array2 based on binning timescale. Since 
        Array 2 is shifted wrt Array 1. Array2 should have equal or finer
        time resolution.
        """
        dt1 = times1[1] - times1[0]
        dt2 = times2[1] - times2[0]

        if dt2 <= dt1:
            self._times1, self._counts1 = times1, counts1
            self._times2, self._counts2 = times2, counts2
            self._dt1, self._dt2 = dt1, dt2
            self._num_shifts = int(np.floor(self._max_dt / dt2))
        else:
            self._times1, self._counts1 = times2, counts2
            self._times2, self._counts2 = times1, counts1
            self._dt1, self._dt2 = dt2, dt1
            self._num_shifts = int(np.floor(self._max_dt / dt1))

    def _check_arrays(self):
        if len(self._times1) != len(self._times2):
            return  # Arrays are already misaligned in size; nothing to do

        if (self._times1[0] != self._times2[0]) or (self._dt1 != self._dt2):
            self._trim_arrays()

    def _trim_arrays(self):
        start_diff = self._times1[0] - self._times2[0]
        if abs(start_diff) <= self._dt2:
            return  # Close enough

        if start_diff < 0:
            new_start = self._times2[self._num_shifts] - self._max_dt
            idx = np.searchsorted(self._times1, new_start)
            if idx > 0:
                self._times1 = self._times1[idx:]
                self._counts1 = self._counts1[idx:]
                self._times2 = self._times2[:-idx]
                self._counts2 = self._counts2[:-idx]
        else:
            new_start = self._times1[0] + self._max_dt
            idx = np.searchsorted(self._times2, new_start) - self._num_shifts - 1
            if idx > 0:
                self._times2 = self._times2[idx:]
                self._counts2 = self._counts2[idx:]
                self._times1 = self._times1[:-idx]
                self._counts1 = self._counts1[:-idx]

    @property
    def num_shifts(self):
        return self._num_shifts

    @num_shifts.setter
    def num_shifts(self, val):
        try:
            int_val = int(val)
        except:
            raise TypeError('num_shifts should be a non-negative integer')
        if int_val < 0:
            raise ValueError('num_shifts should be a non-negative integer')
        if int_val >= self._counts1.size:
            raise ValueError('num_shifts is larger than the array sizes')
        
        self._num_shifts = int_val
        self._shift_array = self._calc_shift_array()
        self._shift_slices = self._calc_shift_slices()
        self._iter = 0

    def _calc_shift_slices(self):
        slices = []
        for shift in self._shift_array:
            if shift < 0:
                # Negative shift: shift array2 backward, 
                # chop off the end of array1 and start of array2
                s1 = slice(0, shift)            # up to -N
                s2 = slice(-shift, None)        # from N to end
            elif shift > 0:
                # Positive shift: shift array2 forward, 
                # chop start of array1 and end of array2
                s1 = slice(shift, None)         # from N to end
                s2 = slice(0, -shift)           # up to -N
            else:
                # No shift
                s1 = slice(None)
                s2 = slice(None)
            slices.append((s1, s2))
        return slices

    @property
    def shift_array(self):
        return self._shift_array   

    def shift(self, shift_idx):
        """Shift array2 relative to array1 given the index into the 
        `shift_array`.
        
        Args:
            shift_idx (int): The index into `shift_array`
        
        Returns:
            (np.array, np.array): The shifted/sliced arrays
        """
        size = self.shift_array.size-1
        if shift_idx < 0:
            raise ValueError('shift_idx must be non-negative')
        if shift_idx > size:
            raise ValueError('shift_idx is too large, must be <= {}'.format(size))

        s1, s2 = self._shift_slices[shift_idx]
        arr1 = [self._times1[s1], self._counts1[s1]]
        arr2 = [self._times2[s2], self._counts2[s2]]
        return arr1, arr2
