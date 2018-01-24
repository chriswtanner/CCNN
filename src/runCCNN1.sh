#!/bin/bash

# params for the CCNN
# -------------------
# list of features to use, where:
# 1=POS; 2=lemma; 3=lemma of depencency parents; 4=char emb.;
# 5=word embeddings (using semantic space vectors)
featureMap=(2)
numLayers=2
numEpochs=10
windowSize=0 # N words to use which are before and after the mention
numNegPerPos=5 # number of negative examples per single pos., during training CCNN
batchSize=128 # number of examples to take it at a time before updates
embeddingsBaseFile="6B.300" # 6B.300") # "840B.300")
dropout=0.0 # dropout for the CCNN
CCNNOpt="adam" # optimizer to use for the CCNN
clusterMethod=("min") # DELETE (should always be MIN)
numFilters=32 # number for filters in the first layer of CCNN
filterMultiplier=2.0 # multiplier rate for # of filters to use on each
# subsequent layer of CCNN (e.g., if prev param was 32, and this value is 2.0,
# then we'll use 32,64,128 units for the 1st 3 layers, respectively)
hddcrpBaseFile="predict.ran" # the file that contains the test mentions' to use
featurePOS="none" # none   onehot   emb_random   emb_glove
posType="none" # none  sum  avg how to combine the embeddings over multiple mention tokens
lemmaType="sum" # "sum" "avg"; how to combine the embeddings over multiple mention tokens
dependencyType="none" # # "sum" "avg"; how to combine the embeddings over multiple mention tokens
charType="none" # "none" "concat" "sum" "avg"; how to combine the char embeddings
SSType="none" # "none" "sum" "avg"; how to combine the embeddings over multiple mention tokens
SSwindowSize=0 # 3 5 7 # represent each word embedding based on its context of windows this size
SSvectorSize=0 #100 400 800) # dimn of the word embedding
SSlog="True" # DELETE THIS
devDir=20 # 2 3 4 5 6 7 8 9 10 11 12 13 14 16 18 19 20 21 22 23 24 25)

# cd /home/christanner/researchcode/DeepCoref/src/
hn=`hostname`

# FEATURE MAP OVERRIDE
if [[ " ${featureMap[*]} " == *"1"* ]]; then
	featurePOS=("emb_glove")
	posType=("sum")
fi
if [[ " ${featureMap[*]} " == *"2"* ]]; then
	lemmaType=("sum")
fi
if [[ " ${featureMap[*]} " == *"3"* ]]; then
	dependencyType=("sum")
fi
if [[ " ${featureMap[*]} " == *"4"* ]]; then
	charType=("concat")
fi
if [[ " ${featureMap[*]} " == *"5"* ]]; then
	SSType=("sum")
	SSwindowSize=(5)
	SSvectorSize=(400)
fi

# FFNN params
FFNNnumEpochs=20 # 5 20
FFNNPosRatio=0.8 # 0.2 0.8
FFNNOpt="adam" # "rms" "adam" "adagrad"
echo $hn
fout="output.txt"
if [ ${hn} = "titanx" ] || [ ${hn} = "Christophers-MacBook-Pro-2" ]
then
	echo "* running runCCNN2 natively"
	./runCCNN2.sh FULL gpu ${numLayers} ${numEpochs} ${windowSize} ${numNegPerPos} ${batchSize} ${embeddingsBaseFile} ${hddcrpBaseFile} ${dropout} ${CCNNOpt} ${numFilters} ${filterMultiplier} ${featurePOS} ${posType} ${lemmaType} ${dependencyType} ${charType} ${SSType} ${SSwindowSize} ${SSvectorSize} ${devDir} ${FFNNnumEpochs} ${FFNNPosRatio} ${FFNNOpt}
else
	qsub -l gpus=1 -o ${fout} runCCNN2.sh FULL gpu ${nl} ${pool} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hdd} ${dr} ${co} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${dt} ${ct} ${st} ${ws2} ${vs} ${sl} ${dd} ${fn} ${fp} ${fo}
fi