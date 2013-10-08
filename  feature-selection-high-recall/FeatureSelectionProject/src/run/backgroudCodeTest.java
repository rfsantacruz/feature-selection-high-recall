package run;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import problems.ClassificationProblem;
import utils.Util;
import weka.attributeSelection.AttributeSelection;
import weka.core.Option;
import weka.core.Utils;
import classifiers.AbstractLinearClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;
import evaluation.Evaluator;
import experiment.ExperimentExecutor;
import experiment.ExperimentReport;
import experiment.IExperimentCommand;

public class backgroudCodeTest implements IExperimentCommand {

	
	@Override
	public List<ExperimentReport> execute(ClassificationProblem cp) {
		
		List<ExperimentReport> result = new ArrayList<ExperimentReport>();
		
		System.out.println(cp.getName());
		Map<String, String[]> param = new HashMap<String, String[]>();
		param.put("-C", new String[]{"-C 0.1", "-C 0.3", "-C 1.0", "-C 1.3"});
		param.put("-B", new String[]{"-B 0.1", "-B 0.3", "-B 1.0", "-B 1.3"});
		
		AbstractLinearClassifier lr = new LogisticRegressionClassifier();
		AbstractLinearClassifier svm = new SVMLinearClassifier();
		AbstractLinearClassifier nb = new NaiveBayesClassifier();
		
		try {
			
			Evaluator ev = new Evaluator(cp);
			ExperimentReport lrReport = ev.crossValidateModel(lr, cp, 10, System.currentTimeMillis(), param);
			System.out.println(lrReport);
			result.add(lrReport);
			/*ExperimentReport svmReport = ev.crossValidateModel(svm, cp, 10, System.currentTimeMillis(), null);
			System.out.println(svmReport);
			ExperimentReport nbReport = ev.crossValidateModel(nb, cp, 10, System.currentTimeMillis(), null);
			System.out.println(nbReport);*/
					
		} catch (Exception e) {
			e.printStackTrace();
		}
	
		return result;
	}
	
	
	public static void main(String[] args) {
		String path = "./data";
		IExperimentCommand cmd = new backgroudCodeTest();
		ExperimentExecutor exe = new ExperimentExecutor();
		List<ExperimentReport> result = exe.executeCommandInFiles(cmd, path);
		utils.Util.saveExperimentReportAsCSV("./results/backgroudtest.csv", result, ",");
	}

	
	

}
