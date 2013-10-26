package evaluation;


import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.beans.AbstractEvaluator;
import classifiers.AbstractLinearClassifier;

import com.google.common.collect.Lists;

import experiment.ExperimentReport;

public class WekaEvaluationWrapper{

	private Evaluation wekaEvaluation ;
	private ClassificationProblem problem;

	public WekaEvaluationWrapper(ClassificationProblem cp){
		this.problem = cp;
	}

	public ExperimentReport crossValidateModel(AbstractLinearClassifier c, ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params ){

		ExperimentReport reportPerf = new ExperimentReport(cp.getName(), c.getClassifierName());

		try{

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
				int trainSize = (int)Math.round(trainAndValid.size() * 0.8);
				Instances train = new Instances(trainAndValid, 0, trainSize);
				Instances valid = new Instances(trainAndValid, trainSize , trainAndValid.size() - train.size());

				if(params != null && !params.isEmpty()){
					List<String> modelSettings = Util.generateModels(Lists.newArrayList(params.values()));
					String optimumSetting = "";
					double erroRate = Double.MAX_VALUE;
					for (String setting : modelSettings) {
						c.resetClassifier();
						//set parameter, train and evaluate
						c.setOptions(Utils.splitOptions(setting));
						c.buildClassifier(train);
						this.evaluateModel(c, valid);
						double currentErrorRate = errorRate();

						if(currentErrorRate  < erroRate){
							erroRate = currentErrorRate;
							optimumSetting = setting;
						}
					}
					c.resetClassifier();
					c.setOptions(Utils.splitOptions(optimumSetting));
				}

				//train with optimum parameters
				c.buildClassifier(trainAndValid);

				//test and report the performance
				this.evaluateModel(c, test);
				reportPerf.setAccuracy(reportPerf.getAccuracy() + ((1-errorRate())/folds));
				reportPerf.setPrecision(reportPerf.getPrecision() + (precision()/folds));
				reportPerf.setRecall(reportPerf.getRecall() + ((recall())/folds));
				reportPerf.setF_measure(reportPerf.getF_measure() + (fMeasure()/folds));
			}



		}catch(Exception e){
			e.printStackTrace();
		}
		return reportPerf;

	}
	
	//wrraped metrics
	
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
	public void evaluateModel(AbstractLinearClassifier c, Instances test) throws Exception {
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
	public double precision() {
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
	public double recall() {
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
