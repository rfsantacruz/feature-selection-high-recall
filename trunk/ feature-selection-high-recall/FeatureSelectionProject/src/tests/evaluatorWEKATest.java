package tests;

import java.util.Arrays;

import problems.ClassificationProblem;
import classifiers.NaiveBayesClassifier;
import evaluation.Evaluator;
import weka.classifiers.evaluation.Evaluation;

public class evaluatorWEKATest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
		NaiveBayesClassifier classifier = new NaiveBayesClassifier();
		
		try {
			System.out.println("weka eval 1");
			Evaluation eval = new Evaluation(cp.getData());
			classifier.buildClassifier(cp.getData());
			eval.evaluateModel(classifier, cp.getData());
			System.out.println(eval.errorRate());
			System.out.println(eval.precision(0));
			System.out.println(eval.recall(0));
			
			System.out.println("weka eval 2");
			classifier.buildClassifier(cp.getData());
			eval.evaluateModel(classifier, cp.getData());
			System.out.println(eval.errorRate());
			System.out.println(eval.precision(0));
			System.out.println(eval.recall(0));
			
			
			Evaluator eval2 = new Evaluator(cp);
			
			System.out.println("My eval 1");
			classifier.buildClassifier(cp.getData());
			eval2.evaluateModel(classifier, cp.getData());
			System.out.println(eval2.errorRate());
			System.out.println(eval2.precision());
			System.out.println(eval2.recall());
			
			System.out.println("My eval 2");
			classifier.buildClassifier(cp.getData());
			eval2.evaluateModel(classifier, cp.getData());
			System.out.println(eval2.errorRate());
			System.out.println(eval2.precision());
			System.out.println(eval2.recall());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		

	}

}
