package classifiers;

import java.util.Arrays;

import problems.ClassificationProblem;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesClassifier extends AbstractLinearClassifier {

	//constructor 
	public NaiveBayesClassifier(){
		this.model = new NaiveBayes();
		classifierName = "NaiveBayes";
	}

	//method to train a naive bayes classifier
	@Override
	public void buildClassifier(Instances data)
			throws Exception {
		model.buildClassifier(data);

	}

	//method to classify a new intance using the naive bayes classifyer
	@Override
	public double classifyInstance(Instance newInstance) throws Exception {
		return model.classifyInstance(newInstance);

	}

	//getters and setters with cast
	public NaiveBayes getModel() {
		return (NaiveBayes)model;
	}
	public void setModel(NaiveBayes model) {
		this.model = model;
	}

	//reset default options
	@Override
	public void resetClassifier() {
		model = new NaiveBayesClassifier();

	}


	//test classifier
	public static void main(String[] args) {
		ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
		NaiveBayesClassifier classifier = new NaiveBayesClassifier();
		try {
			classifier.buildClassifier(cp.getData());
			System.out.println(Arrays.toString(classifier.classifyAll(cp.getData(), null)));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
