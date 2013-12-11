package evaluation;


import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;

import problems.ClassificationProblem;
import utils.Util;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import com.google.common.collect.Lists;

import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelectionFilterFactory;

public class WekaEvaluationWrapper{

	private Evaluation wekaEvaluation ;
	private ClassificationProblem problem;

	public WekaEvaluationWrapper(ClassificationProblem cp){
		this.problem = cp;
	}

	//fast cross validation tuning the model with different metrics and reproting these metrics
	public CrossValidationOutput fastCrossValidationTuneByMetric(AbstractClassifier c, Instances randData, int numOfFeaturesToSelect, 
			EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params, int[][] previousFeature)throws Exception{
		
		//output
		CrossValidationOutput cvo = new CrossValidationOutput(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {

			//fold results
			FoldResult fr = new FoldResult();

			//split: (train + validation) and test
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);


			//split: train and validation
			int trainSize = trainAndValid.size() - test.size(); //(int)Math.round(trainAndValid.size() * 0.8);
			Instances train = new Instances(trainAndValid, 0, trainSize);
			Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

			//perform the feature selection in the train data
			AttributeSelection FeatureSelector = FeatureSelectionFilterFactory.
					getInstance().createFilter(efsa, new FeatureSelectionFactoryParameters(numOfFeaturesToSelect, c, randData ,previousFeature[n]));

			//select feature algorithm invocation
			FeatureSelector.SelectAttributes(train);
			//get attributes indexes
			int[] selectedFeatures = FeatureSelector.selectedAttributes();

			//save selected folds
			fr.addSelectedFeature(ArrayUtils.remove(selectedFeatures,selectedFeatures.length -1));

			//build a filter to remove not selected features
			Remove rm = new Remove();
			rm.setInvertSelection(true);
			rm.setAttributeIndicesArray(selectedFeatures);
			rm.setInputFormat(train);

			//remove not selected features
			train = Filter.useFilter(train, rm);
			valid = Filter.useFilter(valid, rm);
			test = Filter.useFilter(test, rm);
			trainAndValid  = Filter.useFilter(trainAndValid, rm);

			for (EClassificationMetric metric : EClassificationMetric.values()) {
				
				//tune the model
				String optimumSetting = tuneClassifier(c, params, metric,
						train, valid);
				
				//train with optimum parameters
				c.setOptions(Utils.splitOptions(optimumSetting));
				c.buildClassifier(trainAndValid);
				
				this.evaluateModel(c, test);
				double metricReportValue = this.computeTunableMetric(metric);
				
				///workaround to save all metrics
				fr.setOptimalSetting( metric +": " + optimumSetting + " " + (fr.getOptimalSetting() != null ? fr.getOptimalSetting():""));
				fr.setMetricReported(metric, metricReportValue);
				
			}

			cvo.addFoldResult(fr);
		}

		return cvo;

	}

	//this method is the fast cross validation where we just tune the model for one metric and report it
	public CrossValidationOutput fastCrossValidationByMetric(AbstractClassifier c, Instances randData, int numOfFeaturesToSelect, EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params, int[][] previousFeature, EClassificationMetric metric)throws Exception{
		//output
		CrossValidationOutput cvo = new CrossValidationOutput(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {

			//fold results
			FoldResult fr = new FoldResult();

			//split: (train + validation) and test
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);


			//split: train and validation
			int trainSize = trainAndValid.size() - test.size(); //(int)Math.round(trainAndValid.size() * 0.8);
			Instances train = new Instances(trainAndValid, 0, trainSize);
			Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

			//perform the feature selection in the train data
			AttributeSelection FeatureSelector = FeatureSelectionFilterFactory.
					getInstance().createFilter(efsa, new FeatureSelectionFactoryParameters(numOfFeaturesToSelect, c, randData ,previousFeature[n]));

			//select feature algorithm invocation
			FeatureSelector.SelectAttributes(train);
			//get attributes indexes
			int[] selectedFeatures = FeatureSelector.selectedAttributes();

			//save selected folds
			fr.addSelectedFeature(ArrayUtils.remove(selectedFeatures,selectedFeatures.length -1));

			//build a filter to remove not selected features
			Remove rm = new Remove();
			rm.setInvertSelection(true);
			rm.setAttributeIndicesArray(selectedFeatures);
			rm.setInputFormat(train);

			//remove not selected features
			train = Filter.useFilter(train, rm);
			valid = Filter.useFilter(valid, rm);
			test = Filter.useFilter(test, rm);
			trainAndValid  = Filter.useFilter(trainAndValid, rm);

			//tune the model
			String optimumSetting = tuneClassifier(c, params, metric,
					train, valid);

			//train with optimum parameters
			c.setOptions(Utils.splitOptions(optimumSetting));
			c.buildClassifier(trainAndValid);

			this.evaluateModel(c, test);
			double metricReportValue = this.computeTunableMetric(metric);

			fr.setMetricReported(metric, metricReportValue);
			fr.setOptimalSetting(optimumSetting);

			cvo.addFoldResult(fr);
		}

		return cvo;
	}

	//this cross validation is to speed up simulations. it resuses the folds and the selected attributes to reduce the search time
	public CrossValidationOutput fastCrossValidation(AbstractClassifier c, Instances randData, int numOfFeaturesToSelect, EFeatureSelectionAlgorithm efsa, int folds, Map<String,Set<String>> params, int[][] previousFeature)throws Exception{

		//output
		CrossValidationOutput cvo = new CrossValidationOutput(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {

			//fold results
			FoldResult fr = new FoldResult();

			//split: (train + validation) and test
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);


			//split: train and validation
			int trainSize = trainAndValid.size() - test.size(); //(int)Math.round(trainAndValid.size() * 0.8);
			Instances train = new Instances(trainAndValid, 0, trainSize);
			Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

			//perform the feature selection in the train data
			AttributeSelection FeatureSelector = FeatureSelectionFilterFactory.
					getInstance().createFilter(efsa, new FeatureSelectionFactoryParameters(numOfFeaturesToSelect, c, randData ,previousFeature[n]));

			//select feature algorithm invocation
			FeatureSelector.SelectAttributes(train);
			//get attributes indexes
			int[] selectedFeatures = FeatureSelector.selectedAttributes();

			//save selected folds
			fr.addSelectedFeature(ArrayUtils.remove(selectedFeatures,selectedFeatures.length -1));

			//build a filter to remove not selected features
			Remove rm = new Remove();
			rm.setInvertSelection(true);
			rm.setAttributeIndicesArray(selectedFeatures);
			rm.setInputFormat(train);

			//remove not selected features
			train = Filter.useFilter(train, rm);
			valid = Filter.useFilter(valid, rm);
			test = Filter.useFilter(test, rm);
			trainAndValid  = Filter.useFilter(trainAndValid, rm);


			//tune the model to find the optimal parameters given a metric
			String optimumSetting = tuneClassifier(c, params, EClassificationMetric.ACCURACY, train, valid);

			//train with optimum parameters
			c.setOptions(Utils.splitOptions(optimumSetting));
			c.buildClassifier(trainAndValid);

			//test and report the performance
			this.evaluateModel(c, test);

			fr.setOptimalSetting(optimumSetting);
			fr.setMetricReported(EClassificationMetric.ACCURACY, this.computeTunableMetric(EClassificationMetric.ACCURACY));
			fr.setMetricReported(EClassificationMetric.PRECISION, this.computeTunableMetric(EClassificationMetric.PRECISION));
			fr.setMetricReported(EClassificationMetric.RECALL, this.computeTunableMetric(EClassificationMetric.RECALL));
			fr.setMetricReported(EClassificationMetric.FSCORE, this.computeTunableMetric(EClassificationMetric.FSCORE));

			cvo.addFoldResult(fr);
		}

		return cvo;

	}

	//this method tune the model for one metric and report the performance of this metric USING FEATURE SELECTION
	public CrossValidationOutput crossValidateModelByMetric(AbstractClassifier c, AttributeSelection FeatureSelector,ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params, EClassificationMetric metric )throws Exception{


		CrossValidationOutput cvo = new CrossValidationOutput(seed, folds);


		Random rand = new Random(seed); 
		//randomize the data
		Instances randData = new Instances(cp.getData());   
		randData.randomize(rand);

		//if it's necessary stratify to put aproximadetly 
		//the same amount of data of different classes in each fold
		//randData.stratify(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {
			//split: (train + validation) and test
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);


			//split: train and validation
			int trainSize = trainAndValid.size() - test.size(); //(int)Math.round(trainAndValid.size() * 0.8);
			Instances train = new Instances(trainAndValid, 0, trainSize);
			Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

			//perform the feature selection in the train data
			if (FeatureSelector != null) {
				//select feature algorithm invocation
				FeatureSelector.SelectAttributes(train);
				//get attributes indexes
				int[] selectedFeatures = FeatureSelector.selectedAttributes();
				//build a filter to remove not selected features
				Remove rm = new Remove();
				rm.setInvertSelection(true);
				rm.setAttributeIndicesArray(selectedFeatures);
				rm.setInputFormat(train);
				//remove not selected features
				train = Filter.useFilter(train, rm);
				valid = Filter.useFilter(valid, rm);
				test = Filter.useFilter(test, rm);
				trainAndValid  = Filter.useFilter(trainAndValid, rm);
			}	

			//tune the model to find the optimal parameters given a metric
			String optimumSetting = tuneClassifier(c, params, metric, train, valid);

			//train with optimum parameters
			c.setOptions(Utils.splitOptions(optimumSetting));
			c.buildClassifier(trainAndValid);

			//test and report the performance
			this.evaluateModel(c, test);
			double metricReportValue = this.computeTunableMetric(metric);

			FoldResult foldResult = new FoldResult();
			foldResult.setMetricReported(metric, metricReportValue);

			cvo.addFoldResult(foldResult);
		}

		return cvo;
	}

	//This is a cross validation of a the model in the metric accuraccy and rport the perfomance in accuracy, precision, recall and fscore WITHOUT use feature selection
	public CrossValidationOutput crossValidateModel(AbstractClassifier c, ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params ) throws Exception{
		return this.crossValidateModel(c, null, cp, folds, seed, params);
	}

	//This is a cross validation of a model in the metric accuraccy and report the perfomance in accuracy, precision, recall and fscore USING use FEATURE SELECTION
	public CrossValidationOutput crossValidateModel(AbstractClassifier c, AttributeSelection FeatureSelector,ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params ) throws Exception{


		CrossValidationOutput cvo = new CrossValidationOutput(seed, folds);


		Random rand = new Random(seed); 
		//randomize the data
		Instances randData = new Instances(cp.getData());   
		randData.randomize(rand);

		//if it's necessary stratify to put aproximadetly 
		//the same amount of data of different classes in each fold
		//randData.stratify(folds);

		//for each fold
		for (int n = 0; n < folds; n++) {
			//split: (train + validation) and test
			Instances trainAndValid = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);


			//split: train and validation
			int trainSize = trainAndValid.size() - test.size(); //(int)Math.round(trainAndValid.size() * 0.8);
			Instances train = new Instances(trainAndValid, 0, trainSize);
			Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

			//perform the feature selection in the train data
			if (FeatureSelector != null) {
				//select feature algorithm invocation
				FeatureSelector.SelectAttributes(train);
				//get attributes indexes
				int[] selectedFeatures = FeatureSelector.selectedAttributes();
				//build a filter to remove not selected features
				Remove rm = new Remove();
				rm.setInvertSelection(true);
				rm.setAttributeIndicesArray(selectedFeatures);
				rm.setInputFormat(train);
				//remove not selected features
				train = Filter.useFilter(train, rm);
				valid = Filter.useFilter(valid, rm);
				test = Filter.useFilter(test, rm);
				trainAndValid  = Filter.useFilter(trainAndValid, rm);
			}	

			//tune the model to find the optimal parameters given a metric
			String optimumSetting = tuneClassifier(c, params, EClassificationMetric.ACCURACY, train, valid);

			//train with optimum parameters
			c.setOptions(Utils.splitOptions(optimumSetting));
			c.buildClassifier(trainAndValid);

			//test and report the performance
			this.evaluateModel(c, test);
			FoldResult fr = new FoldResult(optimumSetting);
			fr.setMetricReported(EClassificationMetric.ACCURACY, this.computeTunableMetric(EClassificationMetric.ACCURACY));
			fr.setMetricReported(EClassificationMetric.PRECISION, this.computeTunableMetric(EClassificationMetric.PRECISION));
			fr.setMetricReported(EClassificationMetric.RECALL, this.computeTunableMetric(EClassificationMetric.RECALL));
			fr.setMetricReported(EClassificationMetric.FSCORE, this.computeTunableMetric(EClassificationMetric.FSCORE));

			cvo.addFoldResult(fr);
		}

		return cvo;

	}


	//compute metric to tune the model to achieve the highest performance in this metric
	//these metrics are tunable to find the model that maximizes it
	private double computeTunableMetric(EClassificationMetric metric) throws Exception {
		double metricValue = 0;
		switch (metric) {
		case ACCURACY:
			metricValue = this.accuracy();
			break;
		case PRECISION:
			metricValue = this.precision();
			break;
		case RECALL:
			metricValue = this.recall();
			break;
		case FSCORE:
			metricValue = this.fMeasure();
			break;
		default:
			metricValue = this.accuracy();
			break;
		}

		return metricValue;
	}

	//method to tune the model
	private String tuneClassifier(AbstractClassifier classifier,
			Map<String, Set<String>> params, EClassificationMetric metric,
			Instances trainData, Instances validationData) throws Exception {

		//tune the model parameters
		String optimumSetting = "";
		if(params != null && !params.isEmpty()){

			List<String> modelSettings = Util.generateModels(Lists.newArrayList(params.values()));

			double maxMetric = Double.MIN_VALUE;
			for (String setting : modelSettings) {
				//set parameter, train and evaluate
				classifier.setOptions(Utils.splitOptions(setting));
				classifier.buildClassifier(trainData);
				this.evaluateModel(classifier, validationData);
				double currentMetric = this.computeTunableMetric(metric);

				if(currentMetric  > maxMetric){
					maxMetric = currentMetric;
					optimumSetting = setting;
				}
			}

		}
		return optimumSetting;
	}

	//wrraped metrics
	public double accuracy() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return 1 - this.wekaEvaluation.errorRate();
	}


	public double errorRate() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.errorRate();
	}

	public double correct() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.correct();
	}

	public double incorrect() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		return this.wekaEvaluation.incorrect();
	}

	//execute before calculate metrics
	public void evaluateModel(AbstractClassifier c, Instances test) throws Exception {
		this.wekaEvaluation = new Evaluation(this.problem.getData());
		this.wekaEvaluation.evaluateModel(c, test);
	}

	//fmeasure
	public double fMeasure() throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.fMeasure(classValue);
		}

		return retValue/numOfLabels;
	}

	//overload for overrall precision
	public double precision()throws Exception {
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.precision(classValue);
		}

		return retValue/numOfLabels;
	}

	//overload for overrall recall
	public double recall() throws Exception{
		if(this.wekaEvaluation == null)
			throw new Exception("The evaluate model was not called before");

		double retValue = 0.0;
		int numOfLabels = 0;

		int classIndex = this.wekaEvaluation.getHeader().classIndex();
		Attribute att  = this.wekaEvaluation.getHeader().attribute(classIndex);

		Enumeration<String> values = att.enumerateValues();

		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += this.wekaEvaluation.recall(classValue);
		}

		return retValue/numOfLabels;
	}

	public ClassificationProblem getProblem() {
		return problem;
	}





}
