# ======================================================
# ROUTING MODULE (Option D: Fraction State Routing)
# ======================================================

print('routing module imported')

# --- Routing function ---
def ROUT(pcr, q, oldq, flowdir, kx):
    rr = (q * 0.001 * pcr.cellarea()) / (24 * 3600)
    ra = pcr.accuflux(flowdir, rr)
    ra = (1 - kx) * ra + kx * oldq
    return ra

# --- Initialization ---
def init(self, pcr, config):
    self.FlowDir = pcr.readmap(self.inpath + config.get('ROUTING', 'flowdir'))
    try:
        self.kx = pcr.readmap(self.inpath + config.get('ROUTING', 'kx'))
    except:
        self.kx = config.getfloat('ROUTING', 'kx')

# --- Initial conditions ---
def initial(self, pcr, config):
    # total routed flow
    try:
        self.QRAold = config.getfloat('ROUT_INIT', 'QRA_init')
    except:
        try:
            self.QRAold = pcr.readmap(self.inpath + config.get('ROUT_INIT', 'QRA_init'))
        except:
            self.QRAold = 0

    # routed components
    pars = ['RootR', 'RootD', 'Rain', 'Snow', 'Glac', 'Base']
    for i in pars:
        try:
            setattr(self, i + 'RAold', pcr.readmap(self.inpath + config.get('ROUT_INIT', i + 'RA_init')))
            setattr(self, i + 'RA_FLAG', True)
        except:
            try:
                setattr(self, i + 'RAold', config.getfloat('ROUT_INIT', i + 'RA_init'))
                setattr(self, i + 'RA_FLAG', True)
            except:
                setattr(self, i + 'RA_FLAG', False)

    # fraction state at routed outlet (starts neutral)
    self.frac_snow_routed_state = pcr.scalar(0.33)
    self.frac_rain_routed_state = pcr.scalar(0.33)
    self.frac_glac_routed_state = pcr.scalar(0.34)

    # previous routed baseflow (for mass-weighted update)
    self.BaseRAprev = pcr.scalar(0.0)


def dynamic(self, pcr, TotR):

    # --- 1) Route total runoff ---
    Q = self.routing.ROUT(pcr, TotR, self.QRAold, self.FlowDir, self.kx)
    self.QRAold = Q
    self.reporting.reporting(self, pcr, 'QallRAtot', Q)

    if self.mm_rep_FLAG == 1 and self.QTOT_mm_FLAG == 1:
        self.QTOTSubBasinTSS.sample(
            ((Q * 3600 * 24) / pcr.catchmenttotal(pcr.cellarea(), self.FlowDir)) * 1000
        )

    # --- 2) Route baseflow components independently ---
    BsnowRA = self.routing.ROUT(pcr, self.BsnowR, getattr(self, 'BsnowRAold', 0.0), self.FlowDir, self.kx)
    BrainRA = self.routing.ROUT(pcr, self.BrainR, getattr(self, 'BrainRAold', 0.0), self.FlowDir, self.kx)
    BglacRA = self.routing.ROUT(pcr, self.BglacR, getattr(self, 'BglacRAold', 0.0), self.FlowDir, self.kx)

    self.BsnowRAold = BsnowRA
    self.BrainRAold = BrainRA
    self.BglacRAold = BglacRA

    # --- 3) Route total baseflow once ---
    BaseRA = self.routing.ROUT(pcr, self.BaseR, self.BaseRAold, self.FlowDir, self.kx)
    self.BaseRAold = BaseRA

    # --- 4) Compute routed fractions ---
    BtotalRA = pcr.max(BsnowRA + BrainRA + BglacRA, 1e-12)
    frac_snow_routed = BsnowRA / BtotalRA
    frac_rain_routed = BrainRA / BtotalRA
    frac_glac_routed = BglacRA / BtotalRA


    # --- 5) Reporting and QA ---
    # Check mass consistency: BtotalRA should be very close to BaseRA
    self.reporting.reporting(self, pcr, 'BtotalRA', BtotalRA)
    self.reporting.reporting(self, pcr, 'BaseRAtot', BaseRA)

    self.reporting.reporting(self, pcr, 'BsnowRAtot', BsnowRA)
    self.reporting.reporting(self, pcr, 'BrainRAtot', BrainRA)
    self.reporting.reporting(self, pcr, 'BglacRAtot', BglacRA)
    self.reporting.reporting(self, pcr, 'frac_snow_routed', frac_snow_routed)
    self.reporting.reporting(self, pcr, 'frac_rain_routed', frac_rain_routed)
    self.reporting.reporting(self, pcr, 'frac_glac_routed', frac_glac_routed)

    # --- 6) Route other components normally ---
    other_pars = ['RootR', 'RootD', 'Rain', 'Snow', 'Glac']
    for i in other_pars:
        if getattr(self, i + 'RA_FLAG') == 1:
            try:
                ParsRA = self.routing.ROUT(
                    pcr, getattr(self, i + 'R'), getattr(self, i + 'RAold'),
                    self.FlowDir, self.kx
                )
            except:
                ParsRA = self.routing.ROUT(
                    pcr, eval(i + 'R'), getattr(self, i + 'RAold'),
                    self.FlowDir, self.kx
                )
            setattr(self, i + 'RAold', ParsRA)
            self.reporting.reporting(self, pcr, i + 'RAtot', ParsRA)

            if self.mm_rep_FLAG == 1 and getattr(self, 'Q' + i.upper() + '_mm_FLAG') == 1:
                setattr(
                    self,
                    'Q' + i.upper() + 'SubBasinTSS.sample',
                    ((ParsRA * 3600 * 24) /
                     pcr.catchmenttotal(pcr.cellarea(), self.FlowDir)) * 1000
                )

    return Q