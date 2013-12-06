package run;

import java.util.List;

import ExperimentCommands.FeatureSelectionCommand;
import classifiers.ELinearClassifier;

import com.google.common.collect.Lists;

import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;

public class FSSimulation {
	public static void main(String[] args) {

		double start = System.currentTimeMillis();

		//parameters of simulation
		String dataSetFolderpath = "./data";
		String graphOutPutFolderPath = "./results/graphs";
		String csvResultsPath = "./results/featureSelection.csv";

		List<ELinearClassifier> classifiers = Lists.newArrayList(
				ELinearClassifier.LOGISTIC_REGRESSION
				,ELinearClassifier.SVM_LINEAR
				,ELinearClassifier.NAIVE_BAYES
				);


		List<EFeatureSelectionAlgorithm> selectionAlgs = Lists.newArrayList(
				EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK
				,EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK
				,EFeatureSelectionAlgorithm.GAINRATIO_RANK
				//,EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK
				//,EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK
				//,EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET
				//,EFeatureSelectionAlgorithm.FCBF
				//,EFeatureSelectionAlgorithm.RELIFF
				//,EFeatureSelectionAlgorithm.SVMRFE
				//,EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER
				);

		//settinf the simulation
		IExperimentCommand cmd = new FeatureSelectionCommand(classifiers ,selectionAlgs, graphOutPutFolderPath);
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, dataSetFolderpath);

		//save results in csv
		AbstractExperimentReport.saveAll(result, csvResultsPath);

		//time elapsed computation
		System.out.println("elapsed time: " + (System.currentTimeMillis() - start));
	}
}
