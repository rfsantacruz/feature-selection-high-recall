package classifiers;

import java.util.Enumeration;
import java.util.Map;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;


public abstract class AbstractLinearClassifier extends AbstractClassifier {

	//weka model
	protected AbstractClassifier model;
	protected String classifierName;
	
	//classifyall: classify a set of instances
	public double[] classifyAll(Instances newInstances,Map<String, Object> params) throws Exception{
		
		double[] predictedLables = new double[newInstances.size()];
		for (int i = 0; i < newInstances.size(); i++) {
			predictedLables[i] = model.classifyInstance(newInstances.get(i));
		}

		return predictedLables;
	}
	public abstract void resetClassifier();
	
	//identify the classifier
	public String getClassifierName() {
		return classifierName;
	}
	public void setClassifierName(String classifierName) {
		this.classifierName = classifierName;
	}
	
	//override to encapsulate the model option handling
	@Override
	public String[] getOptions() {
		return model.getOptions();
	}
	@Override
	public Enumeration listOptions() {
		return model.listOptions();
	}
	@Override
	public void setOptions(String[] options) throws Exception {
		String[] nop = new String[options.length + 2];
		for (int i = 0; i < options.length; i++) {
			nop[i] = options[i];
		}
		nop[options.length] = "-W ";
		nop[options.length + 1] = "1";
		
		model.setOptions(nop);
	}
	
	
}
