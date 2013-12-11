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
	
	public Map<String,Set<String>> getDefaultClassifiersParameters(ELinearClassifier etype){
		Map<String,Set<String>> param = new HashMap<String,Set<String>>();

		switch (etype) {
		case LOGISTIC_REGRESSION :
			param.put("-C",Sets.newHashSet(" -C 0.01" ,"-C 0.1", "-C 1.0", "-C 10", "-C 100"));
			param.put("-B",Sets.newHashSet(" -B 0.01" ,"-B 0.1", "-B 1.0", "-B 10", "-B 100"));
			break;

		case SVM_LINEAR :
			param.put("-C",Sets.newHashSet(" -C 0.01" ,"-C 0.1", "-C 1.0", "-C 10", "-C 100"));
			param.put("-B",Sets.newHashSet(" -B 0.01" ,"-B 0.1", "-B 1.0", "-B 10", "-B 100"));
			break;

		case NAIVE_BAYES :
			param = null;
			break;
		}

		return param;

	}
	
}
