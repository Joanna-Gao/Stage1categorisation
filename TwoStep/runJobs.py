from os import system, path, getcwd
from otherHelpers import submitJob

dryRun = False
#dryRun = True

myDir = getcwd()
baseDir = '/home/hep/jg4814/CMSSW_10_2_0'
#years = ['2016','2017']

years = ['2016']
intLumi = 35.9

#years = ['2017']
#intLumi = 41.5

# script    = 'diphotonCategorisation.py'
# paramSets = ['max_depth:6']
# #paramSets = [None]
# models    = None
# dataFrame = 'trainTotal.pkl'
# #dataFrame = None
# sigFrame  = None

script    = 'dataSignificances.py'
models    = ['altDiphoModel.model','diphoModel.model']
paramSets = None
dataFrame = 'dataTotal.pkl'
# dataFrame = None
sigFrame  = 'signifTotal.pkl'
# sigFrame  = None

#script    = 'dataMCcheckSidebands.py'
#models    = ['altDiphoModel.model','diphoModel.model']
#paramSets = None
#dataFrame = 'dataTotal.pkl'
#sigFrame  = 'trainTotal.pkl'

#script    = 'dataSignificancesVBF.py'
#models    = [None,'altDiphoModel.model','diphoModel.model']
#paramSets = None
#dataFrame = None
#sigFrame  = None

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
