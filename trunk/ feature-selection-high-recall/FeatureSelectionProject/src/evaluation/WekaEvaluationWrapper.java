package evaluation;


import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import problems.ClassificationProblem;
import utils.Util;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import classifiers.AbstractLinearClassifier;

import com.google.common.collect.Lists;

public class WekaEvaluationWrapper{

	private Evaluation wekaEvaluation ;
	private ClassificationProblem problem;

	public WekaEvaluationWrapper(ClassificationProblem cp){
		this.problem = cp;
	}

	public CrossValidationOutput crossValidateModel(AbstractLinearClassifier c, ClassificationProblem cp, int folds, long seed, Map<String,Set<String>> params ){

		double accuracy = 0;
		double precision = 0;
		double recall = 0;
		double fmeasure = 0;
		
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
				
				String optimumSetting = "";
				if(params != null && !params.isEmpty()){
					List<String> modelSettings = Util.generateModels(Lists.newArrayList(params.values()));
					
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
				}

				//train with optimum parameters
				c.resetClassifier();
				c.setOptions(Utils.splitOptions(optimumSetting));
				c.buildClassifier(trainAndValid);

				//test and report the performance
				this.evaluateModel(c, test);
				accuracy += (this.accuracy()/folds);
				precision += (this.precision()/folds);
				recall += (this.recall()/folds);
				fmeasure += (this.fMeasure()/folds);
			}



		}catch(Exception e){
			e.printStackTrace();
		}
		
		CrossValidationOutput cvo = new CrossValidationOutput(precision, recall, accuracy, fmeasure);
		
		return cvo;

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
