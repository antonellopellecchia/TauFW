#! /usr/bin/env python
# e.g. ./computeScaleFactor.py hists/2018/mutau.root
import os, sys
import ROOT as R

histsPath = sys.argv[1]
print('Reading '+histsPath+'...')
histsFile = R.TFile.Open(histsPath)

histoNameData = 'q_1_SingleMuon_Run2018'
histoNamesMc = ['q_1_ZTT', 'q_1_ZL', 'q_1_EWKT', 'q_1_EWKJ', 'q_1_TopT', 'q_1_TopJ']

# read opposite-sign relaxed-isolation histos
osHistoData = histsFile.Get('os_relaxed/'+histoNameData)
osHistoMc = [ histsFile.Get('os_relaxed/'+name) for name in histoNamesMc ]
# create QCD histogram by subtraction
osHistoQCD = osHistoData.Clone('q_1_QCD')
for histo in osHistoMc: osHistoQCD.Add(histo, -1)
print(osHistoQCD.GetBinContent(1), osHistoQCD.GetBinError(1))

# read same-sign relaxed-isolation histos
ssHistoData = histsFile.Get('ss_relaxed/'+histoNameData)
ssHistoMc = [ histsFile.Get('ss_relaxed/'+name) for name in histoNamesMc ]
# create QCD histogram by subtraction
ssHistoQCD = ssHistoData.Clone('q_1_QCD')
for histo in ssHistoMc: ssHistoQCD.Add(histo, -1)
print(ssHistoQCD.GetBinContent(1), ssHistoQCD.GetBinError(1))

scaleFactorHisto = osHistoQCD.Clone('scale_factor')
scaleFactorHisto.Divide(ssHistoQCD)
print(scaleFactorHisto.GetBinContent(1), scaleFactorHisto.GetBinError(1))

# calculate scale factor by ratio and uncertainty propagation manually, gives same result
scaleFactor = osHistoQCD.GetBinContent(1)/ssHistoQCD.GetBinContent(1)
scaleFactorError = scaleFactor * ((osHistoQCD.GetBinError(1)/osHistoQCD.GetBinContent(1))**2+\
    (ssHistoQCD.GetBinError(1)/ssHistoQCD.GetBinContent(1))**2)**0.5
#print(scaleFactor, scaleFactorError)