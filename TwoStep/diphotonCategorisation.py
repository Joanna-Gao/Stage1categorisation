#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system


from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty

#configure sijofxdkljfhmgckljtomfklc:esting git
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
#parser.add_option('--equalWeights', default=False, action='store_true', help='Alter weights for training so that signal and background have equal sum of weights')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1] #:) 
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
validFrac = 0.1

#get trees from files, put them in data frames
#procfilemap is a dict. 
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'tth':'ttH.root', 'wzh':'VH.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root'}
theProcs = procFileMap.keys()

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems(): #proc,fn are like i,j, key,data
      trainFile   = r.TFile('%s/%s'%(trainDir,fn)) #set up treefile to train from
      if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc) #cheating
      trainTree.SetBranchStatus('nvtx',0) #set values of branches of the training tree. Name of branch, variable value.
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dijet_*',0)
      trainTree.SetBranchStatus('dZ',0)
      trainTree.SetBranchStatus('centralObjectWeight',0)
      trainTree.SetBranchStatus('rho',0)
      trainTree.SetBranchStatus('nvtx',0) #WHY ARE THERE TWO THE SAME
      trainTree.SetBranchStatus('event',0)
      trainTree.SetBranchStatus('lumi',0)
      trainTree.SetBranchStatus('processIndex',0)
      trainTree.SetBranchStatus('run',0)
      trainTree.SetBranchStatus('npu',0)
      trainTree.SetBranchStatus('puweight',0)
      newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
      newTree = trainTree.CloneTree()
      trainFrames[proc] = pd.DataFrame( tree2array(newTree) )
      del newTree #WHYYYY
      del newFile
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.stage1cat>-1.] 
  print 'done basic preselection cuts'
  
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
  trainTotal['diphoWeight'] = trainTotal.apply(diphoWeight,axis=1)
  trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/trainTotal.pkl'%frameDir)
  print 'frame saved as %s/trainTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle(opts.dataFrame)
  print 'Successfully loaded the dataframe'

# testdf1 =  trainTotal.loc[trainTotal['proc'] == 'dipho']
# testdf2 = trainTotal.loc[trainTotal['proc'] == 'gjet']
# print testdf1['proc'], testdf2['proc']
# exit("Plotting not working for now so exit")
sigSumW = np.sum( trainTotal[trainTotal.stage1cat>0.01]['weight'].values )
bkgSumW = np.sum( trainTotal[trainTotal.stage1cat==0]['weight'].values )
print 'sigSumW %.6f'%sigSumW
print 'bkgSumW %.6f'%bkgSumW
print 'ratio %.6f'%(sigSumW/bkgSumW)

#define the indices shuffle (useful to keep this separate so it can be re-used)
theShape = trainTotal.shape[0]
diphoShuffle = np.random.permutation(theShape)
diphoTrainLimit = int(theShape*trainFrac)
diphoValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for diphoton training
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
diphoTW = trainTotal['diphoWeight'].values
diphoAW = trainTotal['altDiphoWeight'].values
diphoFW = trainTotal['weight'].values
diphoM  = trainTotal['CMS_hgg_mass'].values
diphoProc = trainTotal['proc'].values
del trainTotal

diphoX  = diphoX[diphoShuffle] #shuffle indicies to mix up the production modes - going to split into training/test datasets so don't want
diphoY  = diphoY[diphoShuffle] #them all in one.
diphoTW = diphoTW[diphoShuffle]
diphoAW = diphoAW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoM  = diphoM[diphoShuffle]
diphoProc = diphoProc[diphoShuffle]

diphoTrainX,  diphoValidX,  diphoTestX  = np.split( diphoX,  [diphoTrainLimit,diphoValidLimit] ) #splits dataset into training/validation/test
diphoTrainY,  diphoValidY,  diphoTestY  = np.split( diphoY,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainTW, diphoValidTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainAW, diphoValidAW, diphoTestAW = np.split( diphoAW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainFW, diphoValidFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainM,  diphoValidM,  diphoTestM  = np.split( diphoM,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainProc,  diphoValidProc,  diphoTestProc  = np.split( diphoProc,  [diphoTrainLimit,diphoValidLimit] )

#build the background discrimination BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
trainParams = {}
trainParams['objective'] = 'binary:logistic'
trainParams['nthread'] = 1
#trainParams['max_depth'] = 8
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]
'''
print 'about to train diphoton BDT'
diphoModel = xg.train(trainParams, trainingDipho)
print 'done'

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
diphoModel.save_model('%s/diphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/diphoModel%s.model'%(modelDir,paramExt)
'''
'''
#build same thing but with equalised weights
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)
print 'about to train alternative diphoton BDT'
altDiphoModel = xg.train(trainParams, altTrainingDipho)
print 'done'

#save it
altDiphoModel.save_model('%s/altDiphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/altDiphoModel%s.model'%(modelDir,paramExt)


#check performance of each training
diphoPredYxcheck = diphoModel.predict(trainingDipho)
diphoPredY = diphoModel.predict(testingDipho)
print 'Default training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(diphoTrainY, diphoPredYxcheck, sample_weight=diphoTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(diphoTestY, diphoPredY, sample_weight=diphoTestFW) )


altDiphoPredYxcheck = altDiphoModel.predict(trainingDipho)
altDiphoPredY = altDiphoModel.predict(testingDipho)
print 'Alternative training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(diphoTrainY, altDiphoPredYxcheck, sample_weight=diphoTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW) )
'''

# cutfr = 0.5 #fracton to keep for each jackknife iteration
# countvar = 0
# diLente = diphoTestX.shape[0]
# diLentr = diphoTrainX.shape[0]
# rocstest = []
# rocstrain = []
# 
# while countvar <10:
#     # Using default model
#     diphoShufflete = np.random.permutation(diLente)  
#     diphoShuffletr = np.random.permutation(diLentr)
#     
#     TestX = diphoTestX[diphoShufflete][:int(diLente*cutfr)]
#     TestY = diphoTestY[diphoShufflete][:int(diLente*cutfr)]
#     TestFW = diphoTestFW[diphoShufflete][:int(diLente*cutfr)]
#     
#     TrainX = diphoTrainX[diphoShuffletr][:int(diLentr*cutfr)]
#     TrainY = diphoTrainY[diphoShuffletr][:int(diLentr*cutfr)]
#     TrainTW = diphoTrainTW[diphoShuffletr][:int(diLentr*cutfr)]    
#     TrainAW = diphoTrainAW[diphoShuffletr][:int(diLentr*cutfr)]    
#     TrainFW = diphoTrainFW[diphoShuffletr][:int(diLentr*cutfr)]    
# 
#     DefTrainDMatrix = xg.DMatrix(TrainX, label=diphoTrainY[diphoShuffletr][:int(diLentr*cutfr)], weight=TrainTW, feature_names=diphoVars)
#     AltTrainDMatrix = xg.DMatrix(TrainX, label=diphoTrainY[diphoShuffletr][:int(diLentr*cutfr)], weight=TrainAW, feature_names=diphoVars)
#     TestDMatrix = xg.DMatrix(TestX, label=diphoTestY[diphoShufflete][:int(diLente*cutfr)], weight=TestFW, feature_names=diphoVars)
# 
#     diphoPredtrain = altDiphoModel.predict(DefTrainDMatrix)
#     diphoPredtest = altDiphoModel.predict(TestDMatrix)
#     
#     print 'jackknifing (Alt MODEL):'  
#     rocscoretrain = roc_auc_score(TrainY,diphoPredtrain, sample_weight=TrainFW)
#     rocscoretest = roc_auc_score(TestY, diphoPredtest, sample_weight=TestFW)
#     print 'area under training roc curve, iteration', countvar, '= %1.3f'%( rocscoretrain )
#     print 'area under test roc curve, iteration', countvar, '= %1.3f'%( rocscoretest )
#     rocstest.append(rocscoretest)
#     rocstrain.append(rocscoretrain)
#     countvar += 1
# 
# print 'Mean = (train) ', np.mean(rocstrain)
# 
# print 'Mean = (test) ', np.mean(rocstest)
# print 'Standard deviation (test) = ', np.std(rocstest)

#exit("Plotting not working for now so exit")
#make some plots 

plotDir = trainDir.replace('trees','plots')
if not path.isdir(plotDir):
  system('mkdir -p %s'%plotDir)

'''
bkgEff, sigEff, nada = roc_curve(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
plt.figure(1)
plt.plot(bkgEff, sigEff, label='Default Training')
plt.xlabel('Background Efficiency')
plt.ylabel('Signal Efficiency')
#plt.savefig('%s/diphoROC.pdf'%plotDir)

bkgEff, sigEff, nada = roc_curve(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
#plt.figure(2)
plt.plot(bkgEff, sigEff, label='Alternative Training')
plt.plot([0,1], [0,1],'--',color='gray')
# plt.xlabel('Background efficiency')
# plt.ylabel('Signal efficiency')
plt.legend(loc='lower right')
plt.title('ROC Curve for Eta = 0.6, Max Depth = 8')
plt.ylim(0,1)
plt.savefig('%s/BothROC_tuned.pdf'%plotDir)
exit("Plotting not working for now so exit")

plt.figure(3)
xg.plot_importance(diphoModel)
plt.savefig('%s/diphoImportances.pdf'%plotDir)

plt.figure(4)
xg.plot_importance(altDiphoModel)
plt.savefig('%s/altDiphoImportances.pdf'%plotDir)
'''

# Plot individual variables of the event
nOutputBins = 50
leadmvaHist = r.TH1F('leadmvaHist', 'leadmvaHist', nOutputBins, -1, 1)
subleadmvaHist = r.TH1F('subleadmvaHist', 'subleadmvaHist', nOutputBins, -1, 1)
leadptomHist = r.TH1F('leadptomHist', 'leadptomHist', nOutputBins, 0, 2.5)
subleadptomHist = r.TH1F('subleadptomHist', 'subleadptomHist', nOutputBins, 0, 2.5)
leadetaHist = r.TH1F('leadetaHist', 'leadetaHist', nOutputBins, -3.1, 3.1)
subleadetaHist = r.TH1F('subleadetaHist', 'subleadetaHist', nOutputBins, -3.1, 3.1)
CosPhiHist = r.TH1F('CosPhiHist', 'CosPhiHist', nOutputBins, -1, 1)
vtxprobHist = r.TH1F('vtxprobHist', 'vtxprobHist', nOutputBins, 0, 1)
sigmarvHist = r.TH1F('sigmarvHist', 'sigmarvHist', nOutputBins, 0, 0.3)
sigmawvHist = r.TH1F('sigmawvHist', 'sigmawvHist', nOutputBins, 0, 0.3)
# Define bkg hist




#Define a stacked hist
stackHist1 = r.THStack('stackHist1', '')

listHist = [leadmvaHist,subleadmvaHist,leadptomHist,subleadptomHist,
            leadetaHist,subleadetaHist,
            CosPhiHist,vtxprobHist,sigmarvHist,sigmawvHist]
it = 8
#for it in range(3):
theCanv = useSty.setCanvas()
bkgDiphoScoreHist1 = r.TH1F('bkgDiphoScoreHist1', 'bkgDiphoScoreHist', nOutputBins, 0, 0.3)
bkgGjetScoreHist1 = r.TH1F('bkgGjetScoreHist1', 'bkgGjetScoreHist', nOutputBins,0, 0.3)
bkgQCDScoreHist1 = r.TH1F('bkgQCDScoreHist1', 'bkgQCDScoreHist', nOutputBins, 0, 0.3)

bkgDiphoScoreW = 1 * (diphoProc=='dipho')
bkgGjetScoreW = 1 * (diphoProc=='gjet')
bkgQCDScoreW = 1 * (diphoProc=='qcd')
useSty.formatHisto(bkgDiphoScoreHist1)
useSty.formatHisto(bkgGjetScoreHist1)
useSty.formatHisto(bkgQCDScoreHist1)
fill_hist(bkgDiphoScoreHist1, diphoX[:,it], weights=bkgDiphoScoreW)
fill_hist(bkgGjetScoreHist1, diphoX[:,it],  weights=bkgGjetScoreW)
fill_hist(bkgQCDScoreHist1, diphoX[:,it],   weights=bkgQCDScoreW)

sigScoreW1 = 1 * (diphoProc == 'ggh')
useSty.formatHisto(listHist[it])
listHist[it].SetTitle('')
listHist[it].GetXaxis().SetTitle('%s'%diphoVars[it])
fill_hist(listHist[it], diphoX[:,it], weights=sigScoreW1)
listHist[it].Scale(1./listHist[it].Integral())
bkgDiphoScoreHist1.Scale(1./bkgDiphoScoreHist1.Integral())
bkgGjetScoreHist1.Scale(1./bkgGjetScoreHist1.Integral())
bkgQCDScoreHist1.Scale(1./bkgQCDScoreHist1.Integral())
listHist[it].SetFillColor(46)
#bkgDiphoScoreHist1.SetFillColor(34)
#bkgGjetScoreHist1.SetFillColor(31)
#bkgQCDScoreHist1.SetFillColor(40)

#listHist[it].Draw('hist')
#stackHist1.Add(listHist[it])
stackHist1.Add(bkgDiphoScoreHist1)
stackHist1.Add(bkgGjetScoreHist1)
stackHist1.Add(bkgQCDScoreHist1)
stackHist1.Draw('hist')
listHist[it].Draw('hist,same')

useSty.drawCMS()
theCanv.SaveAs('%s/Parameters/%sHistogram.pdf'%(plotDir,diphoVars[it]))



exit("Plotting not working for now so exit")
#draw sig vs background stacked histogram
nOutputBins = 20
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestProc == 'ggh')
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, diphoPredY, weights=sigScoreW)
# bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgDiphoScoreW = diphoTestFW * (diphoTestProc=='dipho')
bkgGjetScoreW = diphoTestFW * (diphoTestProc=='gjet')
bkgQCDScoreW = diphoTestFW * (diphoTestProc=='qcd')

# Define a variable for stack histograms
stackHist = r.THStack('stackHist', '')

# bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
bkgDiphoScoreHist = r.TH1F('bkgDiphoScoreHist', 'bkgDiphoScoreHist', nOutputBins, 0., 1.)
bkgGjetScoreHist = r.TH1F('bkgGjetScoreHist', 'bkgGjetScoreHist', nOutputBins, 0., 1.)
bkgQCDScoreHist = r.TH1F('bkgQCDScoreHist', 'bkgQCDScoreHist', nOutputBins, 0., 1.)
# useSty.formatHisto(bkgScoreHist)
useSty.formatHisto(bkgDiphoScoreHist)
useSty.formatHisto(bkgGjetScoreHist)
useSty.formatHisto(bkgQCDScoreHist)
# bkgScoreHist.SetTitle('')
bkgDiphoScoreHist.SetTitle('')
bkgGjetScoreHist.SetTitle('')
bkgQCDScoreHist.SetTitle('')
# bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
bkgDiphoScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
bkgGjetScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
bkgQCDScoreHist.GetXaxis().SetTitle('Diphoton BDT score')

# fill_hist(bkgScoreHist, altDiphoPredY, weights=bkgScoreW)
fill_hist(bkgDiphoScoreHist, diphoPredY, weights=bkgDiphoScoreW)
fill_hist(bkgGjetScoreHist, diphoPredY, weights=bkgGjetScoreW)
fill_hist(bkgQCDScoreHist, diphoPredY, weights=bkgQCDScoreW)

#apply transformation to flatten ggH
for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    # bkgVal = bkgScoreHist.GetBinContent(iBin)
    bkgDiphoVal = bkgDiphoScoreHist.GetBinContent(iBin)
    bkgGjetVal =  bkgGjetScoreHist.GetBinContent(iBin)
    bkgQCDVal =   bkgQCDScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        # bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
        bkgDiphoScoreHist.SetBinContent(iBin, bkgDiphoVal/sigVal)       
        bkgGjetScoreHist.SetBinContent(iBin, bkgGjetVal/sigVal)
        bkgQCDScoreHist.SetBinContent(iBin, bkgQCDVal/sigVal)
 
    else:
        # bkgScoreHist.SetBinContent(iBin, 0)
        bkgDiphoScoreHist.SetBinContent(iBin, 0)              
        bkgGjetScoreHist.SetBinContent(iBin, 0)
        bkgQCDScoreHist.SetBinContent(iBin, 0)

sigScoreHist.Scale(1./sigScoreHist.Integral())
# bkgScoreHist.Scale(1./bkgScoreHist.Integral())
bkgDiphoScoreHist.Scale(1./bkgDiphoScoreHist.Integral())
bkgGjetScoreHist.Scale(1./bkgGjetScoreHist.Integral())
bkgQCDScoreHist.Scale(1./bkgQCDScoreHist.Integral())
sigScoreHist.SetFillColor(46)
sigScoreHist.Draw('hist')
# bkgScoreHist.SetLineColor(r.kRed)
bkgDiphoScoreHist.SetFillColor(34)
bkgGjetScoreHist.SetFillColor(31)
bkgQCDScoreHist.SetFillColor(40)
bkgDiphoScoreHist.SetMarkerStyle(21)
bkgGjetScoreHist.SetMarkerStyle(21)
bkgQCDScoreHist.SetMarkerStyle(21)
bkgDiphoScoreHist.SetMarkerColor(34)
bkgGjetScoreHist.SetMarkerColor(31)
bkgQCDScoreHist.SetMarkerColor(40)

stackHist.Add(sigScoreHist)
stackHist.Add(bkgDiphoScoreHist)
stackHist.Add(bkgGjetScoreHist)
stackHist.Add(bkgQCDScoreHist) 
# bkgScoreHist.Draw('hist,same')
#bkgDiphoScoreHist.Draw('hist,same')
#bkgGjetScoreHist.Draw('hist,same')
#bkgQCDScoreHist.Draw('hist,same')

stackHist.Draw("hist")
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
#theCanv.SaveAs('%s/outputScores.pdf'%plotDir)
# Logging Y scale
theCanv.SetLogy()
# Add legend
legend = r.TLegend(0.4,0.9,0.4,.9)
legend.AddEntry(sigScoreHist,'ggH Signal','f')
legend.AddEntry(bkgDiphoScoreHist,'Diphoton Background','f')
legend.AddEntry(bkgGjetScoreHist,'GJet Background','f')
legend.AddEntry(bkgQCDScoreHist,'QCD Background','f')
legend.Draw()

theCanv.SaveAs('%s/StackedHistogram_new.pdf'%plotDir)

exit("Plotting not working for now so exit")
#draw sig vs background distribution
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, altDiphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, altDiphoPredY, weights=bkgScoreW)

for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    bkgVal = bkgScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
    else:
        bkgScoreHist.SetBinContent(iBin, 0)
        
sigScoreHist.Scale(1./sigScoreHist.Integral())
bkgScoreHist.Scale(1./bkgScoreHist.Integral())
sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist,same')
useSty.drawCMS()
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/altOutputScores.pdf'%plotDir)
