package run;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import problems.ClassificationProblem;
import utils.Util;
import weka.attributeSelection.ASEvaluation;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.Filter;
import classifiers.AbstractLinearClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;
import evaluation.Evaluator;
import experiment.ExperimentExecutor;
import experiment.ExperimentReport;
import experiment.IExperimentCommand;
import featureSelection.DummyAttributeSelectionAlgorithm;

public class backgroudCodeTest implements IExperimentCommand {


	@Override
	public List<ExperimentReport> execute(ClassificationProblem cp) {

		List<ExperimentReport> result = new ArrayList<ExperimentReport>();

		//feature selection
		Instances featureSelected = this.featureSelection(cp.getData(),3);
		cp.setData(featureSelected);

		//define what parameter optimize to each model
		Map<String, String[]> paramLR = new HashMap<String, String[]>();
		paramLR.put("-C", new String[]{"-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"});
		paramLR.put("-B", new String[]{"-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"});
		
		Map<String, String[]> paramSVM = new HashMap<String, String[]>();
		paramLR.put("-C", new String[]{"-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"});
		paramLR.put("-B", new String[]{"-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"});
		
		
		//create classifier
		AbstractLinearClassifier lr = new LogisticRegressionClassifier();
		AbstractLinearClassifier svm = new SVMLinearClassifier();
		AbstractLinearClassifier nb = new NaiveBayesClassifier();

		try {
			//create the evauator object
			Evaluator ev = new Evaluator(cp);
			
			//cross validate the models
			ExperimentReport lrReport = ev.crossValidateModel(lr, cp, 10, System.currentTimeMillis(), paramLR);
			System.out.println(lrReport);
			result.add(lrReport);
			
			ExperimentReport svmReport = ev.crossValidateModel(svm, cp, 10, System.currentTimeMillis(), paramSVM);
			System.out.println(svmReport);
			result.add(svmReport);
			
			ExperimentReport nbReport = ev.crossValidateModel(nb, cp, 10, System.currentTimeMillis(), null);
			System.out.println(nbReport);
			result.add(nbReport);

			
		} catch (Exception e) {
			e.printStackTrace();
		}

		//return the results of this problem
		return result;
	}


	private Instances featureSelection(Instances trainingInstances, int n) {

		//retrun 
		Instances featureSelected = null;
		try {
			//atribute selection filter
			AttributeSelection attributeSelection = new AttributeSelection();
			//obect to evaluate the attribute and guide the search
			ASEvaluation myfeatureSelection = new DummyAttributeSelectionAlgorithm();
			//choose based on a rank of the attibutes evaluated
			Ranker ranker = new Ranker();
			ranker.setNumToSelect(n);
			//wrap
			attributeSelection.setEvaluator(myfeatureSelection);
			attributeSelection.setSearch(ranker);
			attributeSelection.setInputFormat(trainingInstances);
			//exeute
			featureSelected = Filter.useFilter(trainingInstances, attributeSelection);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//return the data set with just the feature selected
		return featureSelected;
	}


	public static void main(String[] args) {
		String path = "./data";
		IExperimentCommand cmd = new backgroudCodeTest();
		ExperimentExecutor exe = new ExperimentExecutor();
		List<ExperimentReport> result = exe.executeCommandInFiles(cmd, path);
		utils.Util.saveExperimentReportAsCSV("./results/backgroudtest.csv", result, ",");
	}




}
