package run;

import java.util.List;

import org.apache.commons.lang3.time.StopWatch;

import ExperimentCommands.FastFeatureSelectionCommand;
import ExperimentCommands.FastSimulationTuneByDifferentMetrics;
import ExperimentCommands.FeatureSelectionCommand;
import classifiers.ELinearClassifier;

import com.google.common.collect.Lists;

import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;

public class FastFeatureSelectionSimulation {
	public static void main(String[] args) {

		StopWatch stp = new StopWatch();stp.start();

		//parameters of simulation***********
		
		//Paths
		String dataSetFolderpath = "./data";
		String graphOutPutFolderPath = "./results/graphs";
		String csvResultsPath = "./results/featureSelection.csv";
		
		//limit of features to select eg 50 -> run from 0 to 50 features
		int KFeatures = 30; 
		
		//number of folds in cros validation. However remember to change the costant 2.62 in the plots or in the experiment report class
		int folds = 10;
				
		//clasifiers
		List<ELinearClassifier> classifiers = Lists.newArrayList(
				ELinearClassifier.LOGISTIC_REGRESSION
				,ELinearClassifier.SVM_LINEAR
				,ELinearClassifier.NAIVE_BAYES
				);

		// feature selection algorithms
		List<EFeatureSelectionAlgorithm> selectionAlgs = Lists.newArrayList(
				EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK
				,EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK
				,EFeatureSelectionAlgorithm.GAINRATIO_RANK
				,EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK
				//,EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK
				//,EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.FCBF
				//,EFeatureSelectionAlgorithm.RELIFF
				//,EFeatureSelectionAlgorithm.SVMRFE
				//,EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER
				//EFeatureSelectionAlgorithm.HIGH_PRE_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_PRE_LOGLIK_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_EXPECT_APP
				//,EFeatureSelectionAlgorithm.HIGH_REC_LOG_APP
				);
		

		//Simulations type:
		//Fast simulation tuning parameters in accuracy and report in accuracy, precision, fmeasure and recall
		//IExperimentCommand cmd = new FastFeatureSelectionCommand(classifiers ,selectionAlgs, graphOutPutFolderPath, KFeatures, folds);
		//create a fast simulation but to tune a model in different metrics
		IExperimentCommand cmd = new FastSimulationTuneByDifferentMetrics(classifiers ,selectionAlgs, graphOutPutFolderPath, KFeatures, folds);
		
		//execute the simulation
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, dataSetFolderpath);

		//save results in csv
		AbstractExperimentReport.saveAll(result, csvResultsPath);

		//time elapsed computation
		stp.stop();
		
		System.out.println("Total simulation time: (Hours:Minutes:Second.Milisecond): " + stp.toString());
	}
}