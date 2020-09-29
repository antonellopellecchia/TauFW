# Author: Izaak Neutelings (May 2020)
# Description: Simple module to pre-select mutau events
from ROOT import TFile, TTree, TH1D
from ROOT import Math
import numpy as np
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from TauFW.PicoProducer.analysis.utils import deltaPhi


# Inspired by 'Object' class from NanoAODTools.
# Convenient to do so to be able to add MET as 4-momentum to other physics objects using p4()
class Met(Object):
  def __init__(self,event,prefix,index=None):
    self.eta = 0.0
    self.mass = 0.0
    Object.__init__(self,event,prefix,index)


class ModuleMuTau(Module):
  
  def __init__(self,fname,**kwargs):
    self.outfile = TFile(fname,'RECREATE')
    self.default_float = -999.0
    self.default_int = -999
    self.dtype      = kwargs.get('dtype', 'data')
    self.ismc       = self.dtype=='mc'
    self.isdata     = self.dtype=='data'
  
  def beginJob(self):
    """Prepare output analysis tree and cutflow histogram."""
    
    # CUTFLOW HISTOGRAM
    self.cutflow           = TH1D('cutflow','cutflow',25,0,25)
    self.cut_none          = 0
    self.cut_trig          = 1
    self.cut_muon          = 2
    self.cut_muon_veto     = 3
    self.cut_tau           = 4
    self.cut_electron_veto = 5
    self.cut_pair          = 6
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_none,           "no cut"        )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_trig,           "trigger"       )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon,           "muon"          )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon_veto,      "muon     veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_tau,            "tau"           )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_electron_veto,  "electron veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_pair,           "pair"          )
    
    # TREE
    self.tree        = TTree('tree','tree')
    self.pt_1        = np.zeros(1,dtype='f')
    self.eta_1       = np.zeros(1,dtype='f')
    self.q_1         = np.zeros(1,dtype='i')
    self.id_1        = np.zeros(1,dtype='?')
    self.iso_1       = np.zeros(1,dtype='f')
    self.genmatch_1  = np.zeros(1,dtype='f')
    self.decayMode_1 = np.zeros(1,dtype='i')
    self.pt_2        = np.zeros(1,dtype='f')
    self.eta_2       = np.zeros(1,dtype='f')
    self.q_2         = np.zeros(1,dtype='i')
    self.id_2        = np.zeros(1,dtype='i')
    self.anti_e_2    = np.zeros(1,dtype='i')
    self.anti_mu_2   = np.zeros(1,dtype='i')
    self.iso_2       = np.zeros(1,dtype='f')
    self.genmatch_2  = np.zeros(1,dtype='f')
    self.decayMode_2 = np.zeros(1,dtype='i')
    self.m_vis       = np.zeros(1,dtype='f')
    self.genWeight   = np.zeros(1,dtype='f')
    self.tree.Branch('pt_1',         self.pt_1,        'pt_1/F'       )
    self.tree.Branch('eta_1',        self.eta_1,       'eta_1/F'      )
    self.tree.Branch('q_1',          self.q_1,         'q_1/I'        )
    self.tree.Branch('id_1',         self.id_1,        'id_1/O'       )
    self.tree.Branch('iso_1',        self.iso_1,       'iso_1/F'      )
    self.tree.Branch('genmatch_1',   self.genmatch_1,  'genmatch_1/F' )
    self.tree.Branch('decayMode_1',  self.decayMode_1, 'decayMode_1/I')
    self.tree.Branch('pt_2',         self.pt_2,  'pt_2/F'             )
    self.tree.Branch('eta_2',        self.eta_2, 'eta_2/F'            )
    self.tree.Branch('q_2',          self.q_2,   'q_2/I'              )
    self.tree.Branch('id_2',         self.id_2,  'id_2/I'             )
    self.tree.Branch('anti_e_2',     self.anti_e_2,   'anti_e_2/I'    )
    self.tree.Branch('anti_mu_2',    self.anti_mu_2,  'anti_mu_2/I'   )
    self.tree.Branch('iso_2',        self.iso_2, 'iso_2/F'            )
    self.tree.Branch('genmatch_2',   self.genmatch_2,  'genmatch_2/F' )
    self.tree.Branch('decayMode_2',  self.decayMode_2, 'decayMode_2/I')
    self.tree.Branch('m_vis',        self.m_vis, 'm_vis/F'            )
    self.tree.Branch('genWeight',    self.genWeight,   'genWeight/F'  )

    # additional quantities
    self.n_jets = np.zeros(1,dtype='f')
    self.leading_jet_pt = np.zeros(1,dtype='f')
    self.subleading_jet_pt = np.zeros(1,dtype='f')
    self.n_jets_b = np.zeros(1,dtype='f')
    self.leading_jet_b_pt = np.zeros(1,dtype='f')
    self.leading_jet_b_eta = np.zeros(1,dtype='f')
    self.subleading_jet_b_pt = np.zeros(1,dtype='f')
    self.subleading_jet_b_eta = np.zeros(1,dtype='f')
    self.met_pt = np.zeros(1,dtype='f')
    self.met_phi = np.zeros(1,dtype='f')
    self.met_sum_et = np.zeros(1,dtype='f')
    self.puppimet_pt = np.zeros(1,dtype='f')
    self.puppimet_phi = np.zeros(1,dtype='f')
    self.puppimet_sum_et = np.zeros(1,dtype='f')

    self.tree.Branch('n_jets', self.n_jets, 'n_jets/F')
    self.tree.Branch('leading_jet_pt', self.leading_jet_pt, 'leading_jet_pt/F')
    self.tree.Branch('subleading_jet_pt', self.subleading_jet_pt, 'subleading_jet_pt/F')
    self.tree.Branch('n_jets_b', self.n_jets_b, 'n_jets_b/F')
    self.tree.Branch('leading_jet_b_pt', self.leading_jet_b_pt, 'leading_jet_b_pt/F')
    self.tree.Branch('leading_jet_b_eta', self.leading_jet_b_eta, 'leading_jet_b_eta/F')
    self.tree.Branch('subleading_jet_b_pt', self.subleading_jet_b_pt, 'subleading_jet_b_pt/F')
    self.tree.Branch('subleading_jet_b_eta', self.subleading_jet_b_eta, 'subleading_jet_b_eta/F')
    self.tree.Branch('met_pt', self.met_pt, 'met_pt/F')
    self.tree.Branch('met_phi', self.met_phi, 'met_phi/F')
    self.tree.Branch('met_sum_et', self.met_sum_et, 'met_sum_et/F')
    self.tree.Branch('puppimet_pt', self.puppimet_pt, 'puppimet_pt/F')
    self.tree.Branch('puppimet_phi', self.puppimet_phi, 'puppimet_phi/F')
    self.tree.Branch('puppimet_sum_et', self.puppimet_sum_et, 'puppimet_sum_et/F')

    self.r_mu_tau    = np.zeros(1, dtype='f')
    self.mt_1        = np.zeros(1, dtype='f')
    self.mt_2        = np.zeros(1, dtype='f')
    
    self.tree.Branch('r_mu_tau',     self.r_mu_tau, 'r_mu_tau/F'      )
    self.tree.Branch('mt_1',        self.mt_1, 'mt_1/F'            )
    self.tree.Branch('mt_2',        self.mt_2, 'mt_2/F'            )

  
  def endJob(self):
    """Wrap up after running on all events and files"""
    self.outfile.Write()
    self.outfile.Close()
  
  def analyze(self, event):
    """Process event, return True (pass, go to next module) or False (fail, go to next event)."""
    
    # NO CUT
    self.cutflow.Fill(self.cut_none)
    
    # TRIGGER
    if not event.HLT_IsoMu27: return False
    self.cutflow.Fill(self.cut_trig)
    
    # SELECT MUON
    muons = [ ]
    # TODO section 4: extend with a veto of additional muons. Veto muons should have the same quality selection as signal muons (or even looser),
    # but with a lower pt cut, e.g. muon.pt > 15.0
    veto_muons = [ ]

    for muon in Collection(event,'Muon'):
      good_muon = muon.mediumId and muon.pfRelIso04_all < 0.5 and abs(muon.eta) < 2.5
      signal_muon = good_muon and muon.pt > 28.0
      veto_muon   = good_muon and muon.pt > 15.0
      if signal_muon:
        muons.append(muon)
      if veto_muon: # CAUTION: that's NOT an elif here and intended in that way!
        veto_muons.append(muon)
     
    if len(muons) == 0: return False
    self.cutflow.Fill(self.cut_muon)
    # TODO section 4: What should be the requirement to veto events with additional muons?
    if len(veto_muons) >= 2: return False
    self.cutflow.Fill(self.cut_muon_veto)
    
    # SELECT TAU
    # TODO section 6: Which decay modes of a tau should be considered for an analysis? Extend tau selection accordingly
    taus = [ ]
    for tau in Collection(event,'Tau'):
      good_tau = tau.pt > 18.0 and tau.idDeepTau2017v2p1VSe >= 1 and tau.idDeepTau2017v2p1VSmu >= 1 and tau.idDeepTau2017v2p1VSjet >= 1
      if good_tau:
        taus.append(tau)
    if len(taus)<1: return False
    self.cutflow.Fill(self.cut_tau)

    # SELECT ELECTRONS FOR VETO
    # TODO section 4: extend the selection of veto electrons: pt > 15.0,
    # with loose WP of the mva based ID (Fall17 training without isolation),
    # and a custom isolation cut on PF based isolation using all PF candidates.
    electrons = []
    for electron in Collection(event,'Electron'):
      veto_electron = electron.pt > 15.0
      veto_electron = veto_electron and electron.mvaFall17V2noIso_WPL
      veto_electron = veto_electron and electron.pfRelIso03_all < 0.3
      if veto_electron:
        electrons.append(electron)
    if len(electrons) > 0: return False
    self.cutflow.Fill(self.cut_electron_veto)
    
    # PAIR
    # TODO section 4 (optional): the mutau pair is constructed from a muon with highest pt and a tau with highest pt.
    # However, there is also the possibility to select the mutau pair according to the isolation.
    # If you like, you could try to implement mutau pair building algorithm, following the instructions on
    # https://twiki.cern.ch/twiki/bin/view/CMS/HiggsToTauTauWorking2017#Pair_Selection_Algorithm, but using the latest isolation quantities/discriminators
    muon = max(muons,key=lambda p: p.pt)
    tau  = max(taus,key=lambda p: p.pt)
    if muon.DeltaR(tau)<0.4: return False
    self.cutflow.Fill(self.cut_pair)

    # SELECT Jets
    # TODO section 4: Jets are not used directly in our analysis, but it can be good to have a look at least the number of jets (and b-tagged jets) of your selection.
    # Therefore, collect at first jets with pt > 20, |eta| < 4.7, passing loose WP of Pileup ID, and tight WP for jetID.
    # The collected jets are furthermore not allowed to overlap with the signal muon and signal tau in deltaR, so selected them to have deltaR >= 0.5 w.r.t. the signal muon and signal tau.
    # Then, select for this collection "usual" jets, which have pt > 30 in addition, count their number, and store pt & eta of the leading and subleading jet.
    # For b-tagged jets, require additionally DeepFlavour b+bb+lepb tag with medium WP and |eta| < 2.5, count their number, and store pt & eta of the leading and subleading b-tagged jet.
    jets = []
    jets_b = []
    for jet in Collection(event, 'Jet'):
      basic_jet = jet.pt > 20.0 and abs(jet.eta) < 4.7
      basic_jet = basic_jet and jet.puId > 4 # see https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
      basic_jet = basic_jet and jet.jetId > 1 # see https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID
      muon = max(muons,key=lambda p: p.pt)
      tau  = max(taus,key=lambda p: p.pt)
      basic_jet = basic_jet and jet.DeltaR(muon) >= 0.5
      basic_jet = basic_jet and jet.DeltaR(tau) >= 0.5
    
      better_jet = basic_jet and jet.pt > 30.0
      if better_jet: jets.append(jet)

      b_tagged_jet = basic_jet and abs(jet.eta) < 2.5
      b_tagged_jet = b_tagged_jet and jet.btagDeepFlavB > 0.4184 and jet.btagDeepFlavB <  0.2770
      if b_tagged_jet: jets_b.append(jet)

    # CHOOSE MET definition
    # TODO section 4: compare the PuppiMET and (PF-based) MET in terms of mean, resolution and data/expectation agreement of their own distributions and of related quantities
    # and choose one of them for the refinement of Z to tautau selection.
    puppimet = Met(event, 'PuppiMET')
    met = Met(event, 'MET')
    
    # SAVE VARIABLES
    # TODO section 4: extend the variable list with more quantities (also high level ones). Compute at least:
    # - visible pt of the Z boson candidate
    # - best-estimate for pt of Z boson candidate (now including contribution form neutrinos)
    # - transverse mass of the system composed from the muon and MET vectors. Definition can be found in doi:10.1140/epjc/s10052-018-6146-9.
    #   Caution: use ROOT DeltaPhi for difference in phi and check that deltaPhi is between -pi and pi.Have a look at transverse mass with both versions of MET
    # - Dzeta. Definition can be found in doi:10.1140/epjc/s10052-018-6146-9. Have a look at the variable with both versions of MET
    # - Separation in DeltaR between muon and tau candidate
    # - global event quantities like the proper definition of pileup density rho, number of reconstructed vertices,
    # - in case of MC: number of true (!!!) pileup interactions
    self.pt_1[0]        = muon.pt
    self.eta_1[0]       = muon.eta
    self.q_1[0]         = muon.charge
    self.id_1[0]        = muon.mediumId
    self.iso_1[0]       = muon.pfRelIso04_all # keep in mind: the SMALLER the value, the more the muon is isolated
    self.decayMode_1[0] = self.default_int # not needed for a muon
    self.pt_2[0]        = tau.pt
    self.eta_2[0]       = tau.eta
    self.q_2[0]         = tau.charge
    self.id_2[0]        = tau.idDeepTau2017v2p1VSjet
    self.anti_e_2[0]    = tau.idDeepTau2017v2p1VSe
    self.anti_mu_2[0]   = tau.idDeepTau2017v2p1VSmu
    self.iso_2[0]       = tau.rawDeepTau2017v2p1VSjet # keep in mind: the HIGHER the value of the discriminator, the more the tau is isolated
    self.decayMode_2[0] = tau.decayMode
    self.m_vis[0]       = (muon.p4()+tau.p4()).M()

    # jet additional quantities
    self.n_jets[0] = len(jets)
    if len(jets)>0: self.leading_jet_pt[0] = jets[0].pt
    if len(jets)>1: self.subleading_jet_pt[0] = jets[1].pt
    self.n_jets_b[0] = len(jets_b)
    if len(jets_b)>0: self.leading_jet_b_pt[0] = jets_b[0].pt
    if len(jets_b)>0: self.leading_jet_b_eta[0] = jets_b[0].eta
    if len(jets_b)>1: self.subleading_jet_b_pt[0] = jets_b[1].pt
    if len(jets_b)>1: self.subleading_jet_b_eta[0] = jets_b[1].eta

    # met additional quantities
    self.met_pt[0] = met.pt
    self.met_phi[0] = met.phi
    self.met_sum_et[0] = met.sumEt
    self.puppimet_pt[0] = puppimet.pt
    self.puppimet_phi[0] = puppimet.phi
    self.puppimet_sum_et[0] = puppimet.sumEt

    # additional variables useful for QCD bkg estimation
    self.r_mu_tau[0]       = muon.DeltaR(tau)
    self.mt_1[0] = np.sqrt(2*self.pt_1[0]*met.pt*(1-np.cos(deltaPhi(muon.phi,met.phi)) ) )
    self.mt_2[0] = np.sqrt(2*self.pt_2[0]*met.pt*(1-np.cos(deltaPhi(muon.phi,met.phi)) ) )


    if self.ismc:
      self.genmatch_1[0]  = muon.genPartFlav # in case of muons: 1 == prompt muon, 15 == muon from tau decay, also other values available for muons from jets
      self.genmatch_2[0]  = tau.genPartFlav # in case of taus: 0 == unmatched (corresponds then to jet),
                                            #                  1 == prompt electron, 2 == prompt muon, 3 == electron from tau decay,
                                            #                  4 == muon from tau decay, 5 == hadronic tau decay
      self.genWeight[0] = event.genWeight
    self.tree.Fill()
    
    return True
