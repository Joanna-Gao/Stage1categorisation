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
del trainTotal

diphoX  = diphoX[diphoShuffle] #shuffle indicies to mix up the production modes - going to split into training/test datasets so don't want
diphoY  = diphoY[diphoShuffle] #them all in one.
diphoTW = diphoTW[diphoShuffle]
diphoAW = diphoAW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoM  = diphoM[diphoShuffle]

diphoTrainX,  diphoValidX,  diphoTestX  = np.split( diphoX,  [diphoTrainLimit,diphoValidLimit] ) #splits dataset into training/validation/test
diphoTrainY,  diphoValidY,  diphoTestY  = np.split( diphoY,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainTW, diphoValidTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainAW, diphoValidAW, diphoTestAW = np.split( diphoAW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainFW, diphoValidFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainM,  diphoValidM,  diphoTestM  = np.split( diphoM,  [diphoTrainLimit,diphoValidLimit] )

#build the background discrimination BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
trainParams = {}
trainParams['objective'] = 'binary:logistic'
trainParams['nthread'] = 1
trainParams['eta'] = 0.6
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]
print 'about to train diphoton BDT'
diphoModel = xg.train(trainParams, trainingDipho)
print 'done'
'''
#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
diphoModel.save_model('%s/diphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/diphoModel%s.model'%(modelDir,paramExt)
'''

#build same thing but with equalised weights
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)
print 'about to train alternative diphoton BDT'
altDiphoModel = xg.train(trainParams, altTrainingDipho)
print 'done'
'''
#save it
altDiphoModel.save_model('%s/altDiphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/altDiphoModel%s.model'%(modelDir,paramExt)
'''

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
bkgEff, sigEff, nada = roc_curve(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
plt.figure(1)
plt.plot(bkgEff, sigEff, label='Default Training')
plt.xlabel('Background Efficiency')
plt.ylabel('Signal Efficiency')
#plt.savefig('%s/diphoROC.pdf'%plotDir)

bkgEff, sigEff, nada = roc_curve(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
#plt.figure(2)
plt.plot(bkgEff, sigEff, label='Alternative Training')
plt.plot([0,1], [0,1])
# plt.xlabel('Background efficiency')
# plt.ylabel('Signal efficiency')
plt.legend(loc='lower right')
plt.title('ROC Curve for Eta = 0.6, Max Depth = 8')
plt.ylim(0,1)
plt.savefig('%s/BothROC.pdf'%plotDir)
exit("Plotting not working for now so exit")

plt.figure(3)
xg.plot_importance(diphoModel)
plt.savefig('%s/diphoImportances.pdf'%plotDir)

plt.figure(4)
xg.plot_importance(altDiphoModel)
plt.savefig('%s/altDiphoImportances.pdf'%plotDir)

#exit("Plotting not working for now so exit")
#draw sig vs background distribution
nOutputBins = 50
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, diphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, diphoPredY, weights=bkgScoreW)


#apply transformation to flatten ggH
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
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/outputScores.pdf'%plotDir)

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
