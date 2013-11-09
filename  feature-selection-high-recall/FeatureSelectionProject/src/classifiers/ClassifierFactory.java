package classifiers;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;

import weka.classifiers.AbstractClassifier;

public class ClassifierFactory {

	//singleton impl
	private static ClassifierFactory instance;
	public synchronized static ClassifierFactory getInstance(){
		if(instance == null)
			instance = new ClassifierFactory();

		return instance;
	}
	public ClassifierFactory(){}

	//create classifier to be used in the simulations 
	public AbstractClassifier createClassifier(ELinearClassifier EType) {
		AbstractClassifier ret = null;

		switch (EType) {
		case LOGISTIC_REGRESSION :
			ret = new LogisticRegressionClassifier();
			break;

		case SVM_LINEAR :
			ret = new SVMLinearClassifier();
			break;

		case NAIVE_BAYES :
			ret = new NaiveBayesClassifier();
			break;
		}

		return ret;
	}
	
}
