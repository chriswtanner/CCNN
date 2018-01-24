import sys  
import params
import os.path
from ECBParser import *
from ECBHelper import *
from HDDCRPParser import *
from StanParser import *
from CCNN import *
from FFNN import *
from get_coref_metrics import *

# Coreference Resolution System for Events (uses ECB+ corpus)
class CorefEngine:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setCorefEngineParams()

		stoppingPoints = [0.45]
		hddcrp_parsed = None

		# parses the real, actual corpus (ECB's XML files)
		corpus = ECBParser(args)
		helper = ECBHelper(args, corpus, hddcrp_parsed)

		# loads stanford's parsed version of our corpus and aligns it w/
		# our representation -- so we can use their features
		stan = StanParser(args, corpus)
		helper.addStanfordAnnotations(stan)

		# runs CCNN -> FFNN
		if args.testMentions == "hddcrp":
			hddcrp_parsed = HDDCRPParser(args.hddcrpFullFile) # loads HDDCRP's pred or gold mentions file
		elif not args.testMentions.startsWith("ecb"):
			print("ERROR: args.testMentions should be ecbdev, ecbtest, or hddcrp")
			exit(1)

		ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed)
		(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
		ffnnEngine = FFNN(args, corpus, helper, hddcrp_parsed, dev_pairs, dev_preds, testing_pairs, testing_preds)

		ffnnEngine.train()

		for sp in stoppingPoints:
			(predictedClusters, goldenClusters) = ffnnEngine.cluster(sp)
			print("# goldencluster:",str(len(goldenClusters)))
			print("# predicted:",str(len(predictedClusters)))
			if args.testMentions.startsWith("ecb"): # use corpus' gold test set
				(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
				print("FFNN F1 sp:",str(sp),"=",str(conll_f1),"OTHERS:",str(muc_f1),str(bcub_f1),str(ceafe_f1))
			elif args.testMentions == "hddcrp":
				print("FFNN on HDDCRP")
				helper.writeCoNLLFile(predictedClusters, sp)
			else:
				print("ERROR: args.testMentions should be ecbdev, ecbtest, or hddcrp")
				exit(1)






