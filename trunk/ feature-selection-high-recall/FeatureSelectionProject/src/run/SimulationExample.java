package run;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import problems.ClassificationProblem;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import classifiers.AbstractLinearClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;

import com.google.common.collect.Sets;

import evaluation.CrossValidationOutput;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.ClassificationExperimentReport;
import experiment.ExperimentExecutor;
import experiment.IExperimentCommand;
import featureSelection.DummyAttributeSelectionAlgorithm;

public class SimulationExample implements IExperimentCommand {


	@Override
	public List<AbstractExperimentReport> execute(ClassificationProblem cp) {

		List<AbstractExperimentReport> result = new ArrayList<AbstractExperimentReport>();

		//feature selection
		//Instances featureSelected = this.featureSelection(cp.getData(),3);
		//cp.setData(featureSelected);

		//define what parameter each model will cross validate
		HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
		paramLR.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"));
		paramLR.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"));
		
		HashMap<String,Set<String>> paramSVM = new HashMap<String,Set<String>>();
		paramSVM.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"));
		paramSVM.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"));
		//bayes there is no parameters
		
		//create classifier
		AbstractLinearClassifier lr = new LogisticRegressionClassifier();
		AbstractLinearClassifier svm = new SVMLinearClassifier();
		AbstractLinearClassifier nb = new NaiveBayesClassifier();

		try {
			//create the evauator object
			WekaEvaluationWrapper ev = new WekaEvaluationWrapper(cp);
			
			//cross validate the models
			CrossValidationOutput lrcv = ev.crossValidateModel(lr, cp, 10, System.currentTimeMillis(), paramLR);
			System.out.println(lrcv);
			result.add(new ClassificationExperimentReport(lrcv.getPrecision(), lrcv.getRecall(), lrcv.getAccuracy()
					, lrcv.getF_measure(), cp.getName(), lr.getClassifierName()));
			
			CrossValidationOutput svmcv = ev.crossValidateModel(svm, cp, 10, System.currentTimeMillis(), paramSVM);
			System.out.println(svmcv);
			result.add(new ClassificationExperimentReport(svmcv.getPrecision(), svmcv.getRecall(), svmcv.getAccuracy()
					, svmcv.getF_measure(), cp.getName(), svm.getClassifierName()));
			
			CrossValidationOutput nbcv = ev.crossValidateModel(nb, cp, 10, System.currentTimeMillis(), null);
			System.out.println(nbcv);
			result.add(new ClassificationExperimentReport(nbcv.getPrecision(), nbcv.getRecall(), nbcv.getAccuracy()
					, nbcv.getF_measure(), cp.getName(), nb.getClassifierName()));

			
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
			//ASEvaluation myfeatureSelection = new DummySubsetAttributeSelection(); //in case of subset selection
			
			//choose based on a rank of the attibutes evaluated
			Ranker search = new Ranker();
			search.setNumToSelect(n);
			//GreedyStepwise search = new GreedyStepwise(); // in case of subset selection
			
			//wrap
			attributeSelection.setEvaluator(myfeatureSelection);
			attributeSelection.setSearch(search);
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
		double start = System.currentTimeMillis();
		String path = "./data";
		String jarPath = "./data/datasets-UCI.jar";
		IExperimentCommand cmd = new SimulationExample();
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, path);
		//List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInJAR(cmd, jarPath);
		AbstractExperimentReport.saveInFile(result, "./results/backgroudtest.csv");
		System.out.println("elapsed time: " + (System.currentTimeMillis() - start));
	}




}
