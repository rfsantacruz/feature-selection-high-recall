package run;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.collections.ListUtils;

import problems.ClassificationProblem;
import utils.Util;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import classifiers.ClassifierFactory;
import classifiers.ELinearClassifier;
import classifiers.LogisticRegressionClassifier;
import classifiers.NaiveBayesClassifier;
import classifiers.SVMLinearClassifier;

import com.google.common.base.Joiner;
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


public class FeatureSelectionSimulation implements IExperimentCommand{



	private String GraphsPath = "./results/graphs";
	private  List<EFeatureSelectionAlgorithm> selectionAlgs = null;
	private List<ELinearClassifier> classifiers = null;
	private Logger log = Util.getFileLogger(FeatureSelectionSimulation.class.getName(), "./results/logs/log.txt");

	public FeatureSelectionSimulation(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath){

		this.GraphsPath = graphPath;
		this.selectionAlgs = FetSelectionAlgs;
		this.classifiers = classifiers;
	}

	private void savePartialResuls( List<AbstractExperimentReport> results, AbstractExperimentReport... exps){
		for (AbstractExperimentReport exp : exps) {
			exp.plot(GraphsPath);
			results.add(exp);
		}
	}

	private Map<String,Set<String>> setUpClassifiersParameters(ELinearClassifier etype){
		Map<String,Set<String>> param = new HashMap<String,Set<String>>();

		switch (etype) {
		case LOGISTIC_REGRESSION :
			param.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0"));
			param.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0"));
			break;

		case SVM_LINEAR :
			param.put("-C",Sets.newHashSet("-C 0.1", "-C 0.3", "-C 1.0"));
			param.put("-B",Sets.newHashSet("-B 0.1", "-B 0.3", "-B 1.0"));
			break;

		case NAIVE_BAYES :
			param = null;
			break;
		}

		return param;

	}


	@Override
	public List<AbstractExperimentReport> execute(ClassificationProblem cp)  {

		List<AbstractExperimentReport> result = new ArrayList<AbstractExperimentReport>();

		WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);


		for (ELinearClassifier ecl :  this.classifiers){
			
			//get the classifier and the parameters
			AbstractClassifier cl = ClassifierFactory.getInstance().createClassifier(ecl);
			Map<String,Set<String>> param =  this.setUpClassifiersParameters(ecl);

			
			//instatiate reports
			int maxNumFeatures = cp.getNumAttributes() - 1;
			FeatureSelectionExperimentReport exp_acc = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, "Accuracy");
			FeatureSelectionExperimentReport exp_pre = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, "Precision");
			FeatureSelectionExperimentReport exp_rec = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, "Recall");
			FeatureSelectionExperimentReport exp_fm = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, "FMeasure");
			
			//run all selected feature selection algorithm
			for (EFeatureSelectionAlgorithm alg : this.selectionAlgs) {
				
				try {

					for (int n_features = 1; n_features <= cp.getNumAttributes() - 1; n_features++) {

						//get the filter
						AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
								.createFilter(alg, new FeatureSelectionFactoryParameters(n_features, cl, cp.getData()));

						//cross validate
						CrossValidationOutput outlr = eval.crossValidateModel(cl, filter,cp, 10, 10, param);
						
						//collect cross validation measurement
						exp_acc.metricMeanAddValue(alg.name(), outlr.accuracyMean());
						exp_pre.metricMeanAddValue(alg.name(), outlr.precisionMean());
						exp_rec.metricMeanAddValue(alg.name(), outlr.recallMean());
						exp_fm.metricMeanAddValue(alg.name(), outlr.fmeasureMean());
						
						exp_acc.metricStdAddValue(alg.name(), outlr.accuracyStd());
						exp_pre.metricStdAddValue(alg.name(), outlr.precisionStd());
						exp_rec.metricStdAddValue(alg.name(), outlr.recallyStd());
						exp_fm.metricStdAddValue(alg.name(), outlr.fmeasureStd());
						
					}
					System.out.println("Simulation: " +ecl.name() + " and " + alg.name() + " done!" );

				} catch (Exception e) {
					String msg = Joiner.on(" ").skipNulls().join("problem in:",cp.getName(),ecl.name(),alg.name());
					log.log(Level.WARNING,msg , e);
					System.out.println(msg);
				}

			}

			this.savePartialResuls(result, exp_acc, exp_pre, exp_rec, exp_fm);

		}
		
		System.out.println("Simulation in Problem " + cp.getName() + " done!" );

		return result;
	}

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
				,EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK
				,EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK
				,EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET
				,EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET
				,EFeatureSelectionAlgorithm.FCBF
				,EFeatureSelectionAlgorithm.RELIFF
				//,EFeatureSelectionAlgorithm.SVMRFE
				//,EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER
				//,EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER
				);

		//settinf the simulation
		IExperimentCommand cmd = new FeatureSelectionSimulation(classifiers ,selectionAlgs, graphOutPutFolderPath);
		List<AbstractExperimentReport> result = ExperimentExecutor.getInstance().executeCommandInFiles(cmd, dataSetFolderpath);

		//save results in csv
		AbstractExperimentReport.saveAll(result, csvResultsPath);

		//time elapsed computation
		System.out.println("elapsed time: " + (System.currentTimeMillis() - start));
	}

}