package evaluation;


import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import problems.ClassificationProblem;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import classifiers.AbstractLinearClassifier;
import experiment.ExperimentReport;

public class Evaluator extends Evaluation {

	
	public Evaluator(ClassificationProblem cp) throws Exception{
		super(cp.getData());
	}


	public ExperimentReport crossValidateModel(AbstractLinearClassifier c, ClassificationProblem cp, int folds, long seed, Map<String, String[]> params ){

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


				//train and evaluate
				//store param*
				Map<String, String> optimumValue = new HashMap<String, String>();
				if(params != null && params.size() > 0){
					for(String paramKey : params.keySet() ){
						//get the desired tune values
						String[] paramList = params.get(paramKey);
						double erroRate = Double.MAX_VALUE;
						//loop throug the values
						for (String v : paramList) {
							c.resetClassifier();
							//set parameter, train and evaluate
							c.setOptions(Utils.splitOptions(v));
							c.buildClassifier(train);
							this.evaluateModel(c, valid);
							//save the optimum value
							if(errorRate()  < erroRate){
								erroRate = errorRate();
								optimumValue.put(paramKey, v);
							}
						}
					}

					c.resetClassifier();
					//config optimum parameters
					String op = "";
					for (String paramName : optimumValue.keySet()) {
						 op += optimumValue.get(paramName) + " ";
					}
					c.setOptions(Utils.splitOptions(op));
				}
				
				//train with optimum parameters
				c.buildClassifier(trainAndValid);

				//test and report the performance
				this.evaluateModel(c, test);
				reportPerf.setAccuracy(reportPerf.getAccuracy() + ((1-errorRate())/folds));
				reportPerf.setPrecision(reportPerf.getPrecision() + (precision(test)/folds));
				reportPerf.setRecall(reportPerf.getRecall() + ((recall(test))/folds));
				reportPerf.setF_measure(reportPerf.getF_measure() + (fMeasure(test)/folds));

			}

		}catch(Exception e){
			e.printStackTrace();
		}
		return reportPerf;

	}

	@Override
	public void crossValidateModel(Classifier classifier, Instances data,
			int numFolds, Random random, Object... forPredictionsPrinting)
					throws Exception {
		throw new IllegalAccessError("Method does not should to be used");
	}

	@Override
	public void crossValidateModel(String classifierString, Instances data,
			int numFolds, String[] options, Random random) throws Exception {
		throw new IllegalAccessError("Method does not should to be used");
	}


	//overload for overall fmeasure
	public double fMeasure(Instances cp) {
		double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = cp.classIndex();
		Attribute att  = cp.attribute(classIndex);
		Enumeration<String> values = att.enumerateValues();
		
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += super.fMeasure(classValue);
		}
		
		return retValue/numOfLabels;
	}


	//overload for overrall precision
	public double precision(Instances cp) {
		double retValue = 0.0;
		int numOfLabels = 0;
		int classIndex = cp.classIndex();
		Attribute att  = cp.attribute(classIndex);
		Enumeration<String> values = att.enumerateValues();
		
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += super.precision(classValue);
		}
		
		return retValue/numOfLabels;
	}


	//overload for overrall recall
	public double recall(Instances cp) {
		double retValue = 0.0;
		int numOfLabels = 0;
		
		int classIndex = cp.classIndex();
		Attribute att  = cp.attribute(classIndex);
		Enumeration<String> values = att.enumerateValues();
		
		while(values.hasMoreElements()){
			int classValue = att.indexOfValue(values.nextElement());
			numOfLabels++;
			retValue += super.recall(classValue);
		}
		
		return retValue/numOfLabels;
	}

	


}
