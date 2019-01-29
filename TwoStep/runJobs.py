from os import system, path, getcwd
from otherHelpers import submitJob

dryRun = False
#dryRun = True

myDir = getcwd()
baseDir = '/home/hep/jg4814/CMSSW_10_2_0/'
#years = ['2016','2017']

years = ['2016']
intLumi = 35.9

#years = ['2017']
#intLumi = 41.5

#script    = 'diphotonCategorisation.py'
#paramSets = [None] #['eta:0.3','eta:0.4','eta:0.5','eta:0.6', 'eta:0.7','eta:0.8','eta:0.9']
#models    = None
#classModel = None
#dataFrame = 'trainTotal_JECDown01sigma.pkl'#'trainTotal.pkl'
##dataFrame = None
#sigFrame  = None
#treeName = None#'_JECDown01sigma'

#script    = 'nJetCategorisation.py'
#paramSets = ['max_depth:15']
#models    = None
#classModel = None
#dataFrame = 'jetTotal.pkl'
##dataFrame = None
#sigFrame  = None

script    = 'dataSignificances.py'
models    = ['altDiphoModel.model']#'diphoModel.model',
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
paramSets = [None]
classModel = 'jetModel_JECDown01sigma.model'
for params in paramSets:
  if not params: continue
  params = params.split(',')
  name = 'diphoModel'
  for param in params:
    var = param.split(':')[0]
    val = param.split(':')[1]
    name += '__%s_%s'%(var,str(val))
  name += '.model'
  models.append(name)
  models.append(name.replace('dipho','altDipho'))
paramSets = None
dataFrame = 'dataTotal.pkl'
#dataFrame = None
sigFrame  = 'signifTotal.pkl'
#sigFrame  = None

#script    = 'dataMCcheckSidebands.py'
#models    = ['altDiphoModel.model','diphoModel.model']
#classModel = None
#paramSets = None
#dataFrame = 'dataTotal.pkl'
#sigFrame  = 'trainTotal.pkl'

#script    = 'dataSignificancesVBF.py'
#models    = [None,'altDiphoModel.model','diphoModel.model']
#classModel = None
#paramSets = [None,'max_depth:3','max_depth:4','max_depth:5','max_depth:10','eta:0.1','eta:0.5','lambda:0']
#for params in paramSets:
#  if not params: continue
#  params = params.split(',')
#  name = 'diphoModel'
#  for param in params:
#    var = param.split(':')[0]
#    val = param.split(':')[1]
#    name += '__%s_%s'%(var,str(val))
#  name += '.model'
#  models.append(name)
#  models.append(name.replace('dipho','altDipho'))
#paramSets = None
##dataFrame = None
#dataFrame = 'dataTotal.pkl'
##sigFrame  = None
#sigFrame  = 'vbfTotal.pkl'

#script    = 'combinedBDT.py'
#paramSets = None
#models    = [None,'altDiphoModel.model']
#classModel = None
##dataFrame = None
#dataFrame = 'combinedTotal.pkl'
#sigFrame  = None

#script    = 'dataSignificancesVBFcombined.py'
#models = [None,'altDiphoModel.model']
#classModel = None
#paramSets = None
##dataFrame = None
#dataFrame = 'dataTotal.pkl'
##sigFrame  = None
#sigFrame  = 'vbfTotal.pkl'

if __name__=='__main__':
  for year in years:
    jobDir = '%s/Jobs/%s/%s' % (myDir, script.replace('.py',''), year)
    if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
    trainDir  = '%s/%s/trees'%(baseDir,year)
    if 'VBF' in script: trainDir  = '%s/%s/ForVBF/trees'%(baseDir,year) #FIXME
    theCmd = 'python %s -t %s '%(script, trainDir)
    if dataFrame: 
      theCmd += '-d %s/%s/frames/%s '%(baseDir, year, dataFrame) 
    if sigFrame: 
      theCmd += '-s %s/%s/frames/%s '%(baseDir, year, sigFrame) 
    if intLumi: 
      theCmd += '--intLumi %s '%intLumi
    if classModel: 
      theCmd += '--className %s '%classModel
    if paramSets and models:
      exit('ERROR do not expect both parameter set options and models. Exiting..')
    elif paramSets: 
      for params in paramSets:
        fullCmd = theCmd 
        if params: fullCmd += '--trainParams %s '%params
        submitJob( jobDir, fullCmd, params=params, dryRun=dryRun )
    elif models:
      for model in models:
        fullCmd = theCmd
        if model: fullCmd += '-m %s '%model
        submitJob( jobDir, fullCmd, model=model, dryRun=dryRun )
