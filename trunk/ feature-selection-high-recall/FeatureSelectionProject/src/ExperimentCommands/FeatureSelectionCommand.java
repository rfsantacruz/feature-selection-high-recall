package ExperimentCommands;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.collections.ListUtils;
import org.apache.commons.lang3.time.DateUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.lang3.time.StopWatch;

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
import evaluation.EClassificationMetric;
import evaluation.WekaEvaluationWrapper;
import experiment.AbstractExperimentReport;
import experiment.ExperimentExecutor;
import experiment.FeatureSelectionExperimentReport;
import experiment.IExperimentCommand;
import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelectionFilterFactory;


public class FeatureSelectionCommand implements IExperimentCommand{

	private String GraphsPath;
	private  List<EFeatureSelectionAlgorithm> selectionAlgs = null;
	private List<ELinearClassifier> classifiers = null;
	private Logger log = Util.getFileLogger(FeatureSelectionCommand.class.getName(), "./results/logs/log.txt");
	private int KmaxFeatures;
	private StopWatch timer;
	private int folds;

	public FeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath){
		this(classifiers,FetSelectionAlgs,graphPath,Integer.MAX_VALUE, 10);
	}
	
	public FeatureSelectionCommand(List<ELinearClassifier> classifiers, List<EFeatureSelectionAlgorithm> FetSelectionAlgs, String graphPath, int kMaxFeatures, int folds ){

		this.GraphsPath = graphPath;
		this.selectionAlgs = FetSelectionAlgs;
		this.classifiers = classifiers;
		this.KmaxFeatures = kMaxFeatures;
		timer = new StopWatch();timer.start();
		this.folds = folds;
	}


	@Override
	public List<AbstractExperimentReport> execute(ClassificationProblem cp)  {
		
		System.out.println("> problem: " + cp.getName());
		
		List<AbstractExperimentReport> result = new ArrayList<AbstractExperimentReport>();

		WekaEvaluationWrapper eval = new WekaEvaluationWrapper(cp);


		for (ELinearClassifier ecl :  this.classifiers){
			
			System.out.println(">> classifier: " + ecl.name());
			long classifierTime = timer.getTime();
			
			//get the classifier and the parameters
			AbstractClassifier cl = ClassifierFactory.getInstance().createClassifier(ecl);
			Map<String,Set<String>> param =  ClassifierFactory.getInstance().getDefaultClassifiersParameters(ecl);

			//instatiate reports
			int maxNumFeatures = KmaxFeatures < cp.getNumAttributes() - 1 ? KmaxFeatures : cp.getNumAttributes() - 1;
			FeatureSelectionExperimentReport exp_acc = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.ACCURACY.name());
			FeatureSelectionExperimentReport exp_pre = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures, folds, EClassificationMetric.PRECISION.name());
			FeatureSelectionExperimentReport exp_rec = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures,  folds, EClassificationMetric.RECALL.name());
			FeatureSelectionExperimentReport exp_fm = new FeatureSelectionExperimentReport(ecl.name(), cp.getName(), maxNumFeatures,  folds, EClassificationMetric.FSCORE.name());
			
			//run all selected feature selection algorithm
			for (EFeatureSelectionAlgorithm alg : this.selectionAlgs) {
				
				System.out.println(">>> feature selection alg: " + alg.name());
				long fetSeleTimer = timer.getTime();
				
				try {

					for (int n_features = 1; n_features <= maxNumFeatures; n_features++) {
						
						
						//get the filter
						AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
								.createFilter(alg, new FeatureSelectionFactoryParameters(n_features, cl, cp.getData()));

						//cross validate
						CrossValidationOutput outlr = eval.crossValidateModel(cl, filter,cp, folds, System.currentTimeMillis(), param);
						
						//collect cross validation measurement
						exp_acc.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.ACCURACY));
						exp_pre.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.PRECISION));
						exp_rec.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.RECALL));
						exp_fm.metricMeanAddValue(alg.name(), outlr.metricMean(EClassificationMetric.FSCORE));
						
						exp_acc.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.ACCURACY));
						exp_pre.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.PRECISION));
						exp_rec.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.RECALL));
						exp_fm.metricStdAddValue(alg.name(), outlr.metricSTD(EClassificationMetric.FSCORE));
						
						
						
					}
					System.out.println(">>> " + alg.name() + " done! " + "! Elapsed time(HH:mm:ss.S): " + DurationFormatUtils.formatDuration((timer.getTime() - fetSeleTimer), "HH:mm:ss.S") );

				} catch (Exception e) {
					String msg = Joiner.on(" ").skipNulls().join("problem in:",cp.getName(),ecl.name(),alg.name());
					log.log(Level.WARNING,msg , e);
					System.out.println(msg);
				}

			}

			this.savePartialResuls(result, exp_acc, exp_pre, exp_rec, exp_fm);
			
			System.out.println(">> " + ecl.name() + " done! Elapsed time(HH:mm:ss.S): " 
			+ DurationFormatUtils.formatDuration(timer.getTime() - classifierTime, "HH:mm:ss.S") );
		}
		
		System.out.println(">Problem " + cp.getName() + " done! Acutal Simulation time(HH:mm:ss.S): " + timer.toString() );
		
		return result;
	}
	
	
	private void savePartialResuls( List<AbstractExperimentReport> results, AbstractExperimentReport... exps){
		for (AbstractExperimentReport exp : exps) {
			exp.plot(GraphsPath);
			results.add(exp);
		}
	}
}
