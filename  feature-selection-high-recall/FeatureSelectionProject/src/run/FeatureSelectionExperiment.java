package run;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import problems.ClassificationProblem;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import classifiers.LogisticRegressionClassifier;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import evaluation.CrossValidationOutput;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.FeatureSelectionExperimentReport;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelectionFilterFactory;


public class FeatureSelectionExperiment implements IExperimentCommand{

	private static final String GraphsPath = "./results/graphs";
	private static List<AbstractClassifier> classifiers = null;
	
	private static final List<EFeatureSelectionAlgorithm> selectionAlgs = Lists.newArrayList(
			EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK, 
			EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK,
			EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET);

	@Override
	public List<AbstractExperimentReport> execute(ClassificationProblem cp)  {

		List<AbstractExperimentReport> result = new ArrayList<AbstractExperimentReport>();

		WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);

		//logistic regression intanciation
		AbstractClassifier lr = new LogisticRegressionClassifier();
		HashMap<String,Set<String>> paramLR = new HashMap<String,Set<String>>();
		paramLR.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0"));
		paramLR.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0"));

		try {

			int maxNumFeatures = cp.getNumAttributes() - 1;
			Map<String, List<Double>> alg2accuracy = new HashMap<String, List<Double>>();
			Map<String, List<Double>> alg2precision = new HashMap<String, List<Double>>();
			Map<String, List<Double>> alg2recall = new HashMap<String, List<Double>>();
			Map<String, List<Double>> alg2fmeasure = new HashMap<String, List<Double>>();

			for (EFeatureSelectionAlgorithm alg : selectionAlgs) {

				alg2accuracy.put(alg.name(), new ArrayList<Double>());
				alg2precision.put(alg.name(), new ArrayList<Double>());
				alg2recall.put(alg.name(), new ArrayList<Double>());
				alg2fmeasure.put(alg.name(), new ArrayList<Double>());
				
				for (int n_features = 1; n_features <= cp.getNumAttributes() - 1; n_features++) {

					//logistic regression
					AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
							.createFilter(alg, new FeatureSelectionFactoryParameters(n_features, lr, cp.getData()));
					
					CrossValidationOutput outlr = eval.crossValidateModel(lr, filter,cp, 10, 10, paramLR);
					alg2accuracy.get(alg.name()).add(outlr.getAccuracy());
					alg2precision.get(alg.name()).add(outlr.getPrecision());
					alg2recall.get(alg.name()).add(outlr.getRecall());
					alg2fmeasure.get(alg.name()).add(outlr.getF_measure());
				}

			}
			
			FeatureSelectionExperimentReport exp_acc = 
					new FeatureSelectionExperimentReport(alg2accuracy, "logisticRegression", cp.getName(), maxNumFeatures, "Accuracy");
			
			FeatureSelectionExperimentReport exp_pre = 
					new FeatureSelectionExperimentReport(alg2precision, "logisticRegression", cp.getName(), maxNumFeatures, "Precision");
			
			FeatureSelectionExperimentReport exp_rec = 
					new FeatureSelectionExperimentReport(alg2recall, "logisticRegression", cp.getName(), maxNumFeatures, "Recall");
			
			FeatureSelectionExperimentReport exp_fm = 
					new FeatureSelectionExperimentReport(alg2fmeasure, "logisticRegression", cp.getName(), maxNumFeatures, "Fmeasure");
			
			
			exp_acc.saveInFile(GraphsPath);
			exp_pre.saveInFile(GraphsPath);
			exp_rec.saveInFile(GraphsPath);
			exp_fm.saveInFile(GraphsPath);
			
			result.add(exp_acc);
			result.add(exp_pre);
			result.add(exp_rec);
			result.add(exp_fm);
			
		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.println("Simulation in Problem " + cp.getName() + " done!" );

		return result;
	}

	public static void main(String[] args) {
		double start = System.currentTimeMillis();
		String path = "./data";
		IExperimentCommand cmd = new FeatureSelectionExperiment();
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, path);
		//AbstractExperimentReport.saveInFile(result, "./results/featureSelection.csv");
		System.out.println("elapsed time: " + (System.currentTimeMillis() - start));
	}

}
