#!/bin/bash
export PYTHONIOENCODING=UTF-8

# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"

stoppingPoints=(0.51)

if [ ${me} = "ctanner" ]
then
	echo "[ ON BROWN NETWORK ]"
	baseDir=${brownDir}
	refDir=${refDirBrown}
	if [ ${hn} = "titanx" ]
	then
		echo "*   ON TITAN!"
		export CUDA_HOME=/usr/local/cuda/
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
		export PATH=${CUDA_HOME}/bin:${PATH}
	else
		echo "*   ON THE GRID!"
		source ~/researchcode/DeepCoref/venv/bin/activate
		export CUDA_HOME=/contrib/projects/cuda8.0
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
		export PATH=${CUDA_HOME}/bin:${PATH}
	fi 
fi

scriptDir=${baseDir}"src/"
refDir=${scriptDir}"reference-coreference-scorers-8.01/"
corpusPath=${baseDir}"data/ECB_$1/"
replacementsFile=${baseDir}"data/replacements.txt"
charEmbeddingsFile=${baseDir}"data/charRandomEmbeddings.txt"
allTokens=${baseDir}"data/allTokensFull.txt"

hddcrpBaseFile=$9
hddcrpFullFile=${baseDir}"data/"${hddcrpBaseFile}".WD.semeval.txt" # MAKE SURE THIS IS WHAT YOU WANT (gold vs predict)
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"
dataDir=${baseDir}"data/"
resultsDir=${baseDir}"results/"

# glove params
embeddingsBaseFile=$8
gloveOutput=${baseDir}"data/gloveEmbeddings."${embeddingsBaseFile}".txt"

stoplistFile=${baseDir}"data/stopwords.txt"

# additional coref engine params
mentionsFile=${baseDir}"data/goldTruth_events.txt"
embeddingsFile=${gloveOutput}
embeddingsType="type"
useECBTest=true
device=$2
numLayers=$3
numEpochs=$4
windowSize=$5
numNegPerPos=$6
batchSize=$7
dropout=${10}
CCNNOpt=${11}
numFilters=${12}
filterMultiplier=${13}

# CCNN features
featurePOS=${14}
posType=${15}
posEmbeddingsFile=${baseDir}"data/posEmbeddings100.txt"
lemmaType=${16}
dependencyType=${17}
charType=${18}
SSType=${19}
SSwindowSize=${20}
SSvectorSize=${21}
SSlog="True"

devDir=${22}
FFNNnumEpochs=${23}
FFNNPosRatio=${24}
FFNNOpt=${25}

stanOutputDir=${baseDir}"data/stanford_output/"

# constants
shuffleTraining="False"
echo "------------------------"
echo "         params         "
echo "------------------------"
echo "---- generic params ----"
echo "corpus:" $1
echo "useECBTest:" ${useECBTest}
echo "stoplistFile:" $stoplistFile
echo "resultsDir:" ${resultsDir}
echo "dataDir:" ${dataDir}
echo "device:" ${device}
echo "numLayers:" $numLayers
echo "poolType:" $poolType
echo "unified token-formatting file:" ${replacementsFile}
echo "stitchMentions:" $stitchMentions
echo "mentionsFile:" $mentionsFile
echo "embeddingsBaseFile:" $embeddingsBaseFile
echo "embeddingsFile:" $embeddingsFile
echo "embeddingsType:" $embeddingsType
echo "numEpochs:" $numEpochs
echo "verbose:" $verbose
echo "windowSize:" $windowSize
echo "shuffleTraining:" $shuffleTraining
echo "numNegPerPos:" $numNegPerPos
echo "batchSize:" $batchSize
echo "hddcrpBaseFile:" $hddcrpBaseFile
echo "hddcrpFullFile:" $hddcrpFullFile
echo "dropout:" $dropout
echo "CCNNOpt:" $CCNNOpt
echo "clusterMethod:" $clusterMethod
echo "numFilters:" $numFilters
echo "filterMultiplier:" $filterMultiplier
echo "stanOutputDir:" $stanOutputDir
echo "featurePOS:" $featurePOS
echo "posType:" $posType
echo "posEmbeddingsFile:" $posEmbeddingsFile
echo "lemmaType:" $lemmaType
echo "dependencyType:" $dependencyType
echo "charEmbeddingsFile:" $charEmbeddingsFile
echo "charType:" $charType
echo "SSType:" $SSType
echo "SSwindowSize:" $SSwindowSize
echo "SSvectorSize:" $SSvectorSize
echo "SSlog:" $SSlog
echo "devDir:" $devDir
echo "FFNNnumEpochs:" $FFNNnumEpochs
echo "FFNNPosRatio:" $FFNNPosRatio
echo "FFNNOpt:" $FFNNOpt
echo "------------------------"

cd $scriptDir

python3 -u CorefEngine.py --resultsDir=${resultsDir} --dataDir=${dataDir} \
--useECBTest=${useECBTest} \
--stoplistFile=${stoplistFile} \
--device=${device} \
--numLayers=${numLayers} --poolType=${poolType} --corpusPath=${corpusPath} \
--replacementsFile=${replacementsFile} \
--stitchMentions=${stitchMentions} --mentionsFile=${mentionsFile} \
--embeddingsBaseFile=${embeddingsBaseFile} --embeddingsFile=${embeddingsFile} \
--embeddingsType=${embeddingsType} --numEpochs=${numEpochs} --verbose=${verbose} \
--windowSize=${windowSize} --shuffleTraining=${shuffleTraining} --numNegPerPos=${numNegPerPos} \
--batchSize=${batchSize} \
--hddcrpBaseFile=${hddcrpBaseFile} --hddcrpFullFile=${hddcrpFullFile} \
--dropout=${dropout} \
--CCNNOpt=${CCNNOpt} \
--clusterMethod=${clusterMethod} \
--numFilters=${numFilters} --filterMultiplier=${filterMultiplier} \
--stanOutputDir=${stanOutputDir} \
--featurePOS=${featurePOS} --posType=${posType} --posEmbeddingsFile=${posEmbeddingsFile} \
--lemmaType=${lemmaType} \
--dependencyType=${dependencyType} \
--charEmbeddingsFile=${charEmbeddingsFile} \
--charType=${charType} \
--SSType=${SSType} \
--SSwindowSize=${SSwindowSize} \
--SSvectorSize=${SSvectorSize} \
--SSlog=${SSlog} \
--devDir=${devDir} \
--FFNNnumEpochs=${FFNNnumEpochs} \
--FFNNPosRatio=${FFNNPosRatio} \
--FFNNOpt=${FFNNOpt}

if [ "$useECBTest" = false ] ; then
	cd ${refDir}
	goldFile=${baseDir}"data/gold.WD.semeval.txt" # gold.NS.WD.semeval.txt"
	shopt -s nullglob

	for sp in "${stoppingPoints[@]}"
	do
		f=${baseDir}"results/"${hddcrpBaseFile}"_nl"${numLayers}"_pool"${poolType}"_ne"${numEpochs}"_ws"${windowSize}"_neg"${numNegPerPos}"_bs"${batchSize}"_sFalse_e"${embeddingsBaseFile}"_dr"${dropout}"_co"${CCNNOpt}"_cm"${clusterMethod}"_nf"${numFilters}"_fm"${filterMultiplier}"_fp"${featurePOS}"_pt"${posType}"_lt"${lemmaType}"_dt"${dependencyType}"_ct"${charType}"_st"${SSType}"_ws2"${SSwindowSize}"_vs"${SSvectorSize}"_sl"${SSlog}"_dd"${devDir}"_fn"${FFNNnumEpochs}"_fp"${FFNNPosRatio}"_fo"${FFNNOpt}"_sp"${sp}".txt"
		muc=`./scorer.pl muc ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
		bcub=`./scorer.pl bcub ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
		ceafe=`./scorer.pl ceafe ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
		sum=`echo ${muc}+${bcub}+${ceafe} | bc`
		avg=`echo "scale=2;$sum/3.0" | bc`
		echo "CoNLLF1:" ${f} ${avg} "OTHERS:" ${muc} ${bcub} ${ceafe}
		rm -rf ${f}
	done
fi