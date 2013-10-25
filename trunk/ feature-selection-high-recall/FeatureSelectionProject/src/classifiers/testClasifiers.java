package classifiers;

import problems.ClassificationProblem;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

public class testClasifiers {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		try {
			
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			LogisticRegressionClassifier lr = new LogisticRegressionClassifier();
			SVMLinearClassifier svm = new SVMLinearClassifier();
			NaiveBayes nb = new NaiveBayes();
			
			lr.buildClassifier(cp.getData());
			svm.buildClassifier(cp.getData());
			nb.buildClassifier(cp.getData());
			Evaluation ev = new Evaluation(cp.getData());
			
			ev.evaluateModel(lr, cp.getData());
			System.out.println("Logistic regression => Acuraccy:" + (1 - ev.errorRate()));
			
			ev.evaluateModel(svm, cp.getData());
			System.out.println("SVM => Acuraccy:" + (1 - ev.errorRate()));
			
			ev.evaluateModel(nb, cp.getData());
			System.out.println("Naive bayes => Acuraccy:" + (1 - ev.errorRate()));
			
			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	

}
