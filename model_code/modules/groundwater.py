# The Spatial Processes in HYdrology (SPHY) model:
# A spatially distributed hydrological model 
# Copyright (C) 2013-2019  FutureWater
# Email: sphy@futurewater.nl
#
# Authors (alphabetical order):
# P. Droogers, J. Eekhout, W. Immerzeel, S. Khanal, A. Lutz, G. Simons, W. Terink
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

print('groundwater module imported')

#-Function to calculate groundwater recharge
def GroundWaterRecharge(pcr, deltagw, gwrecharge, subperc, glacperc):
    gwseep = (1 - pcr.exp(-1 / deltagw)) * (subperc + glacperc)
    gwrecharge = (pcr.exp(-1 / deltagw) * gwrecharge) + gwseep
    return gwrecharge

#-Function to calculate baseflow
def BaseFlow(pcr, gw, baser, gwrecharge, basethresh, alphagw):
    baser = pcr.ifthenelse(gw <= basethresh, 0, (baser * pcr.exp(-alphagw) + gwrecharge * (1 - pcr.exp(-alphagw))))
    return baser

#-Function to calculate the groundwater height, taken from the bottom of the gw layer (zero reference)
def HLevel(pcr, Hgw, alphagw, gwrecharge, yield_gw):
    Hgw = (Hgw * pcr.exp(-alphagw)) + ((gwrecharge * (1 - pcr.exp(-alphagw))) / (800 * yield_gw * alphagw))
    return Hgw

#-init groundwater processes
def init(self, pcr, config):
    pars = ['GwDepth','GwSat','deltaGw','BaseThresh','alphaGw','YieldGw']
    for i in pars:
        try:
            setattr(self, i, pcr.readmap(self.inpath + config.get('GROUNDW_PARS',i)))
        except:
            setattr(self, i, config.getfloat('GROUNDW_PARS',i))

# --- Initialize groundwater state variables ---
def initial(self, pcr, config):
    # Initial groundwater recharge
    try:
        self.GwRecharge = config.getfloat('GROUNDW_INIT', 'GwRecharge')
    except:
        self.GwRecharge = pcr.readmap(self.inpath + config.get('GROUNDW_INIT', 'GwRecharge'))

    # Initial baseflow
    try:
        self.BaseR = config.getfloat('GROUNDW_INIT', 'BaseR')
    except:
        self.BaseR = pcr.readmap(self.inpath + config.get('GROUNDW_INIT', 'BaseR'))

    # Initial groundwater storage
    try:
        self.Gw = config.getfloat('GROUNDW_INIT', 'Gw')
    except:
        self.Gw = pcr.readmap(self.inpath + config.get('GROUNDW_INIT', 'Gw'))

    # Initial groundwater level
    try:
        self.H_gw = config.getfloat('GROUNDW_INIT', 'H_gw')
    except:
        self.H_gw = pcr.readmap(self.inpath + config.get('GROUNDW_INIT', 'H_gw'))

    self.H_gw = pcr.max(
        (self.RootDepthFlat + self.SubDepthFlat + self.GwDepth) / 1000 - self.H_gw, 0
    )

    # --- Initialize routing-related groundwater attributes ---
    self.BsnowRA_FLAG = True
    self.BrainRA_FLAG = True
    self.BglacRA_FLAG = True
    # If you don't already have these:
    self.GwSnow = 0.0
    self.GwRain = 0.0
    self.GwGlac = 0.0
    self.BsnowRAold = 0.0
    self.BrainRAold = 0.0
    self.BglacRAold = 0.0

def dynamic(self, pcr, ActSubPerc, GlacPerc, SubPercSnow, SubPercRain):

    # --- 1) Total delayed groundwater recharge (existing) ---
    self.GwRecharge = self.groundwater.GroundWaterRecharge(
        pcr, self.deltaGw, self.GwRecharge, ActSubPerc, GlacPerc
    )

    # --- 2) Component-wise delayed recharge ---
    self.GwRecharge_snow = self.groundwater.GroundWaterRecharge(
        pcr, self.deltaGw, getattr(self, 'GwRecharge_snow', 0.0), SubPercSnow, 0.0
    )

    self.GwRecharge_rain = self.groundwater.GroundWaterRecharge(
        pcr, self.deltaGw, getattr(self, 'GwRecharge_rain', 0.0), SubPercRain, 0.0
    )

    self.GwRecharge_glac = self.groundwater.GroundWaterRecharge(
        pcr, self.deltaGw, getattr(self, 'GwRecharge_glac', 0.0), 0.0, GlacPerc
    )

    den = pcr.max(self.GwRecharge_snow + self.GwRecharge_rain + self.GwRecharge_glac, 1e-6)
    frac_snow_delayed = self.GwRecharge_snow / den
    frac_rain_delayed = self.GwRecharge_rain / den
    frac_glac_delayed = self.GwRecharge_glac / den

    # --- 3) Update total groundwater storage ---
    self.Gw += self.GwRecharge
    self.Gw = pcr.max(self.Gw, 0.0)

    # Add component recharge to their storages
    self.GwSnow += self.GwRecharge_snow
    self.GwRain += self.GwRecharge_rain
    self.GwGlac += self.GwRecharge_glac

    # Numerical safety
    self.Gw = pcr.max(self.Gw, 0.0)
    self.GwSnow = pcr.max(self.GwSnow, 0.0)
    self.GwRain = pcr.max(self.GwRain, 0.0)
    self.GwGlac = pcr.max(self.GwGlac, 0.0)

    # --- 5) Compute total baseflow ---
    self.BaseR = self.groundwater.BaseFlow(
        pcr, self.Gw, self.BaseR, self.GwRecharge, self.BaseThresh, self.alphaGw
    )
    self.BaseR *= (1 - self.openWaterFrac)

    # --- 6) Partition outflow by stored fractions ---
    den_storage = pcr.max(self.GwSnow + self.GwRain + self.GwGlac, 1e-9)
    frac_snow_storage = self.GwSnow / den_storage
    frac_rain_storage = self.GwRain / den_storage
    frac_glac_storage = self.GwGlac / den_storage

    BaseR_snow = frac_snow_storage * self.BaseR
    BaseR_rain = frac_rain_storage * self.BaseR
    BaseR_glac = frac_glac_storage * self.BaseR

    self.frac_snow_storage = frac_snow_storage
    self.frac_rain_storage = frac_rain_storage
    self.frac_glac_storage = frac_glac_storage

    # --- 7) Update total GW storage and tracer mass after baseflow outflow ---
    self.Gw -= self.BaseR
    self.Gw = pcr.max(self.Gw, 0.0)

    # Remove component outflow from storages
    self.GwSnow = pcr.max(self.GwSnow - BaseR_snow, 0.0)
    self.GwRain = pcr.max(self.GwRain - BaseR_rain, 0.0)
    self.GwGlac = pcr.max(self.GwGlac - BaseR_glac, 0.0)

    # --- 9) Expose baseflow components ---
    self.BsnowR = BaseR_snow
    self.BrainR = BaseR_rain
    self.BglacR = BaseR_glac
    self.BsumR = BaseR_snow + BaseR_rain + BaseR_glac
    self.BsnowRA_FLAG = self.BrainRA_FLAG = self.BglacRA_FLAG = True

    # --- 10) Reporting ---
    self.reporting.reporting(self, pcr, 'TotGwRechargeF', self.GwRecharge)
    self.reporting.reporting(self, pcr, 'TotBaseRF', self.BaseR)
    self.reporting.reporting(self, pcr, 'BaseR_snow', BaseR_snow)
    self.reporting.reporting(self, pcr, 'BaseR_rain', BaseR_rain)
    self.reporting.reporting(self, pcr, 'BaseR_glac', BaseR_glac)

    self.reporting.reporting(self, pcr, 'TracerSnow', self.GwSnow)
    self.reporting.reporting(self, pcr, 'TracerRain', self.GwRain)
    self.reporting.reporting(self, pcr, 'TracerGlac', self.GwGlac)

    self.reporting.reporting(self, pcr, 'frac_snow_out', self.frac_snow_storage)
    self.reporting.reporting(self, pcr, 'frac_rain_out', self.frac_rain_storage)
    self.reporting.reporting(self, pcr, 'frac_glac_out', self.frac_glac_storage)
    self.reporting.reporting(self, pcr, 'StorGroundW', self.Gw * (1 - self.openWaterFrac))

    # --- 11) Groundwater level ---
    self.H_gw = self.groundwater.HLevel(
        pcr, self.H_gw, self.alphaGw, self.GwRecharge, self.YieldGw
    )
    self.reporting.reporting(
        self, pcr, 'GWL',
        ((self.SubDepthFlat + self.RootDepthFlat + self.GwDepth) / 1000 - self.H_gw) * -1
    )
